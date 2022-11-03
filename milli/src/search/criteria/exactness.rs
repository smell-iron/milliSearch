use std::convert::TryFrom;
use std::mem::take;

use itertools::{Combinations, Itertools};
use log::debug;
use roaring::{MultiOps, RoaringBitmap};

use crate::search::criteria::{
    resolve_phrase, resolve_query_tree, Context, Criterion, CriterionParameters, CriterionResult,
};
use crate::search::query_tree::{Operation, PrimitiveQueryPart};
use crate::{absolute_from_relative_position, FieldId, Result};

pub struct Exactness<'t> {
    ctx: &'t dyn Context<'t>,
    query_tree: Option<Operation>,
    state: Option<State>,
    bucket_candidates: RoaringBitmap,
    parent: Box<dyn Criterion + 't>,
    query: Vec<ExactQueryPart>,
    cache: Option<ExactWordsCombinationCache>,
}

impl<'t> Exactness<'t> {
    pub fn new(
        ctx: &'t dyn Context<'t>,
        parent: Box<dyn Criterion + 't>,
        primitive_query: &[PrimitiveQueryPart],
    ) -> heed::Result<Self> {
        let mut query: Vec<_> = Vec::with_capacity(primitive_query.len());
        for part in primitive_query {
            query.push(ExactQueryPart::from_primitive_query_part(ctx, part)?);
        }

        Ok(Exactness {
            ctx,
            query_tree: None,
            state: None,
            bucket_candidates: RoaringBitmap::new(),
            parent,
            query,
            cache: None,
        })
    }
}

impl<'t> Criterion for Exactness<'t> {
    #[logging_timer::time("Exactness::{}")]
    fn next(&mut self, params: &mut CriterionParameters) -> Result<Option<CriterionResult>> {
        // remove excluded candidates when next is called, instead of doing it in the loop.
        if let Some(state) = self.state.as_mut() {
            state.difference_with(params.excluded_candidates);
        }
        loop {
            debug!("Exactness at state {:?}", self.state);

            match self.state.as_mut() {
                Some(state) if state.is_empty() => {
                    // reset state
                    self.state = None;
                    self.query_tree = None;
                }
                Some(state) => {
                    let (candidates, state) =
                        resolve_state(self.ctx, take(state), &self.query, &mut self.cache)?;
                    self.state = state;

                    return Ok(Some(CriterionResult {
                        query_tree: self.query_tree.clone(),
                        candidates: Some(candidates),
                        filtered_candidates: None,
                        bucket_candidates: Some(take(&mut self.bucket_candidates)),
                    }));
                }
                None => match self.parent.next(params)? {
                    Some(CriterionResult {
                        query_tree: Some(query_tree),
                        candidates,
                        filtered_candidates,
                        bucket_candidates,
                    }) => {
                        let mut candidates = match candidates {
                            Some(candidates) => candidates,
                            None => {
                                resolve_query_tree(self.ctx, &query_tree, params.wdcache)?
                                    - params.excluded_candidates
                            }
                        };

                        if let Some(filtered_candidates) = filtered_candidates {
                            candidates &= filtered_candidates;
                        }

                        match bucket_candidates {
                            Some(bucket_candidates) => self.bucket_candidates |= bucket_candidates,
                            None => self.bucket_candidates |= &candidates,
                        }

                        self.state = Some(State::new(candidates));
                        self.query_tree = Some(query_tree);
                    }
                    Some(CriterionResult {
                        query_tree: None,
                        candidates,
                        filtered_candidates,
                        bucket_candidates,
                    }) => {
                        return Ok(Some(CriterionResult {
                            query_tree: None,
                            candidates,
                            filtered_candidates,
                            bucket_candidates,
                        }));
                    }
                    None => return Ok(None),
                },
            }
        }
    }
}

#[derive(Debug)]
enum State {
    /// Extract the documents that have an attribute that contains exactly the query.
    ExactAttribute(RoaringBitmap),
    /// Extract the documents that have an attribute that starts with exactly the query.
    AttributeStartsWith(RoaringBitmap),
    /// Rank the remaining documents by the number of exact words contained.
    ExactWords(RoaringBitmap),
    Remainings(Vec<RoaringBitmap>),
}

impl State {
    fn new(candidates: RoaringBitmap) -> Self {
        Self::ExactAttribute(candidates)
    }

    fn difference_with(&mut self, lhs: &RoaringBitmap) {
        match self {
            Self::ExactAttribute(candidates)
            | Self::AttributeStartsWith(candidates)
            | Self::ExactWords(candidates) => *candidates -= lhs,
            Self::Remainings(candidates_array) => {
                candidates_array.iter_mut().for_each(|candidates| *candidates -= lhs);
                candidates_array.retain(|candidates| !candidates.is_empty());
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::ExactAttribute(candidates)
            | Self::AttributeStartsWith(candidates)
            | Self::ExactWords(candidates) => candidates.is_empty(),
            Self::Remainings(candidates_array) => {
                candidates_array.iter().all(RoaringBitmap::is_empty)
            }
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::Remainings(vec![])
    }
}
#[logging_timer::time("Exactness::{}")]
fn resolve_state(
    ctx: &dyn Context,
    state: State,
    query: &[ExactQueryPart],
    cache: &mut Option<ExactWordsCombinationCache>,
) -> Result<(RoaringBitmap, Option<State>)> {
    use State::*;
    match state {
        ExactAttribute(mut allowed_candidates) => {
            let mut candidates = RoaringBitmap::new();
            if let Ok(query_len) = u8::try_from(query.len()) {
                let attributes_ids = ctx.searchable_fields_ids()?;
                for id in attributes_ids {
                    if let Some(attribute_allowed_docids) =
                        ctx.field_id_word_count_docids(id, query_len)?
                    {
                        let mut attribute_candidates_array =
                            attribute_start_with_docids(ctx, id, query)?;
                        attribute_candidates_array.push(attribute_allowed_docids);

                        candidates |= intersection_of(attribute_candidates_array.iter().collect());
                    }
                }

                // only keep allowed candidates
                candidates &= &allowed_candidates;
                // remove current candidates from allowed candidates
                allowed_candidates -= &candidates;
            }

            Ok((candidates, Some(AttributeStartsWith(allowed_candidates))))
        }
        AttributeStartsWith(mut allowed_candidates) => {
            let mut candidates = RoaringBitmap::new();
            let attributes_ids = ctx.searchable_fields_ids()?;
            for id in attributes_ids {
                let attribute_candidates_array = attribute_start_with_docids(ctx, id, query)?;
                candidates |= intersection_of(attribute_candidates_array.iter().collect());
            }

            // only keep allowed candidates
            candidates &= &allowed_candidates;
            // remove current candidates from allowed candidates
            allowed_candidates -= &candidates;
            Ok((candidates, Some(ExactWords(allowed_candidates))))
        }
        ExactWords(mut allowed_candidates) => {
            let owned_cache = if let Some(cache) = cache.take() {
                cache
            } else {
                compute_combinations(ctx, query)?
            };

            let mut candidates_array = owned_cache.combinations.clone();
            for candidates in candidates_array.iter_mut() {
                *candidates &= &allowed_candidates;
                allowed_candidates -= &*candidates;
            }
            let all_exact_candidates = candidates_array.pop().unwrap();

            candidates_array.insert(0, allowed_candidates);
            *cache = Some(owned_cache);

            Ok((all_exact_candidates, Some(Remainings(candidates_array))))
        }
        // pop remainings candidates until the emptiness
        Remainings(mut candidates_array) => {
            let candidates = candidates_array.pop().unwrap_or_default();
            if !candidates_array.is_empty() {
                Ok((candidates, Some(Remainings(candidates_array))))
            } else {
                Ok((candidates, None))
            }
        }
    }
}

fn attribute_start_with_docids(
    ctx: &dyn Context,
    attribute_id: FieldId,
    query: &[ExactQueryPart],
) -> heed::Result<Vec<RoaringBitmap>> {
    let mut attribute_candidates_array = Vec::new();
    // start from attribute first position
    let mut pos = absolute_from_relative_position(attribute_id, 0);
    for part in query {
        use ExactQueryPart::*;
        match part {
            Synonyms(synonyms) => {
                let mut synonyms_candidates = RoaringBitmap::new();
                for word in synonyms {
                    let wc = ctx.word_position_docids(word, pos)?;
                    if let Some(word_candidates) = wc {
                        synonyms_candidates |= word_candidates;
                    }
                }
                attribute_candidates_array.push(synonyms_candidates);
                pos += 1;
            }
            Phrase(phrase) => {
                for word in phrase {
                    if let Some(word) = word {
                        let wc = ctx.word_position_docids(word, pos)?;
                        if let Some(word_candidates) = wc {
                            attribute_candidates_array.push(word_candidates);
                        }
                    }
                    pos += 1;
                }
            }
        }
    }

    Ok(attribute_candidates_array)
}

#[inline(never)]
fn intersection_of(mut rbs: Vec<&RoaringBitmap>) -> RoaringBitmap {
    rbs.sort_unstable_by_key(|rb| rb.len());
    roaring::MultiOps::intersection(rbs.into_iter())
}

#[derive(Debug, Clone)]
pub enum ExactQueryPart {
    Phrase(Vec<Option<String>>),
    Synonyms(Vec<String>),
}

impl ExactQueryPart {
    fn from_primitive_query_part(
        ctx: &dyn Context,
        part: &PrimitiveQueryPart,
    ) -> heed::Result<Self> {
        let part = match part {
            PrimitiveQueryPart::Word(word, _) => {
                match ctx.synonyms(word)? {
                    Some(synonyms) => {
                        let mut synonyms: Vec<_> = synonyms
                            .into_iter()
                            .filter_map(|mut array| {
                                // keep 1 word synonyms only.
                                match array.pop() {
                                    Some(word) if array.is_empty() => Some(word),
                                    _ => None,
                                }
                            })
                            .collect();
                        synonyms.push(word.clone());
                        ExactQueryPart::Synonyms(synonyms)
                    }
                    None => ExactQueryPart::Synonyms(vec![word.clone()]),
                }
            }
            PrimitiveQueryPart::Phrase(phrase) => ExactQueryPart::Phrase(phrase.clone()),
        };

        Ok(part)
    }
}

struct ExactWordsCombinationCache {
    // index 0 is only 1 word
    combinations: Vec<RoaringBitmap>,
}

fn compute_combinations(
    ctx: &dyn Context,
    query: &[ExactQueryPart],
) -> Result<ExactWordsCombinationCache> {
    let number_of_part = query.len();
    let mut parts_candidates_array = Vec::with_capacity(number_of_part);
    for part in query {
        let mut candidates = RoaringBitmap::new();
        use ExactQueryPart::*;
        match part {
            Synonyms(synonyms) => {
                for synonym in synonyms {
                    if let Some(synonym_candidates) = ctx.word_docids(synonym)? {
                        candidates |= synonym_candidates;
                    }
                }
            }
            // compute intersection on pair of words with a proximity of 0.
            Phrase(phrase) => {
                candidates |= resolve_phrase(ctx, phrase)?;
            }
        }
        parts_candidates_array.push(candidates);
    }
    let combinations = ComputeCombinations::new(parts_candidates_array).finish();

    Ok(ExactWordsCombinationCache { combinations })
}

/// This structure is used to implement the equivalent of a single function called `compute_combinations`.
///
/// Given a list of bitmaps `b0,b1,...,bn` , it computes another list of bitmaps `X0,X1,...,Xn`
/// where `Xi` contains all the integers that are contained by exactly `i+1` bitmaps.
///
/// The implementation is split into two parts. In the first part, implemented by the `new` function, we build
/// a table (called Levels) containing all the possible combinations of `b0,b1,...,bn`.
///
/// For example, with the bitmaps `b0,b1,b2,b3`, the table should look like this:
/// ```text
/// Level 0: (this contains all the combinations of 1 bitmap)
///     // What follows are lists of intersection of bitmaps asscociated with the index of their last component
///     // There may be multiple lists associated with the same index, that's okay.
///     0: [b0]
///     1: [b1]
///     2: [b2]
///     3: [b3]
/// Level 1: (combinations of 2 bitmaps)
///     1: [b0&b1]
///     2: [b0&b2, b1&b2]
///     3: [b0&b3, b1&b3, b2&b3]
/// Level 2: (combinations of 3 bitmaps)
///     2: [b0&b1&b2]
///     3: [b0&b2&b3, b1&b2&b3]
/// Level 3: (combinations of 4 bitmaps)
///     3: [b0&b1&b2&b3]
/// ```
///
/// These levels are built one by one from the content of the preceding level.
/// For example, to create Level 2, we look at each line of Level 1, for example:
/// ```text
/// 2: [b0&b2, b1&b2]
/// ```
/// And then for each intersection `bx&by` in this list and for each i > 2, we compute `bx&by&bi`
/// and add it the list in Level 3 with the index `i` (if it is not empty):
/// ```text
/// 3: [b0&b2&b3, b1&b2&b3]
/// 4: [b0&b2&b4, b1&b2&b4]
/// 5: [b0&b2&b5, b1&b2&b5]
/// etc.
/// ```
///
/// You can look at the insta-snapshot test `compute_combinations_4` below to see what this looks like
/// for 4 bitmaps.
///
/// After all the Levels are created, we are ready to compute `X0,X1,...Xn`.
/// This is done by the `finish()` method and works as follows.
///
/// We iterate over the Levels in reverse order (starting from the highest one) and compute the
/// union of all of its bitmaps.
///
/// So on the first iteration, we look at Level N and compute the union of everything it contains,
/// which gives us `Xn`. Since `Xi` should contain the elements that are contained by
/// *exactly* `i+1` bitmaps, we need to keep track of which elements were already contained by `Xi+1`
/// before computing `Xi`. This is done by building a "forbidden" bitmap, let's call it `Fi`.
///
/// On the second iteration, we have already computed `Xn` and we have `Fn = Xn`. Again, we compute the union
/// of all the bitmaps inside Level N-1. Then we subtract `Fn` from it. This gives us `Xn-1`. We update the
/// list of forbidden bitmaps such that `Fn-1 = Fn | Xn-1`. Repeat until reaching Level 0.
#[derive(Debug)]
struct ComputeCombinations {
    levels: Vec<Vec<(RoaringBitmap, usize)>>,
}
impl ComputeCombinations {
    #[allow(clippy::explicit_counter_loop, clippy::needless_range_loop)]
    fn new(parts_candidates_array: Vec<RoaringBitmap>) -> Self {
        let nbr_parts = parts_candidates_array.len();
        let base_level: Vec<(RoaringBitmap, usize)> = parts_candidates_array
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, candidates)| (candidates, i))
            .collect();
        if nbr_parts == 1 {
            return Self { levels: vec![base_level] };
        }
        let mut levels = vec![base_level];
        let mut last_level = 0;

        for _ in 2..=nbr_parts {
            let mut new_level = vec![];
            for (base_combination, last_part_index) in levels[last_level].iter() {
                for new_last_part_index in last_part_index + 1..nbr_parts {
                    let new_combination =
                        base_combination & &parts_candidates_array[new_last_part_index];
                    if !new_combination.is_empty() {
                        new_level.push((new_combination, new_last_part_index))
                    }
                }
            }
            levels.push(new_level);
            last_level += 1;
        }

        ComputeCombinations { levels }
    }
    fn finish(self) -> Vec<RoaringBitmap> {
        let mut combinations = vec![];
        let mut forbidden = RoaringBitmap::new();
        for level in self.levels.into_iter().rev() {
            let mut unioned = MultiOps::union(level.into_iter().map(|x| x.0));
            unioned -= &forbidden;
            forbidden |= &unioned;
            combinations.push(unioned)
        }
        combinations
    }
}

#[cfg(test)]
mod tests {
    use roaring::RoaringBitmap;

    use super::ComputeCombinations;
    use crate::snapshot_tests::display_bitmap;

    fn print_compute_combinations(x: &ComputeCombinations) -> String {
        let mut s = String::new();
        for (i, level) in x.levels.iter().enumerate() {
            s.push_str(&format!("Level {}:\n", i + 1));
            for (bitmap, last) in level {
                s.push_str(&format!("    {last} {}\n", &display_bitmap(&bitmap)));
            }
        }
        s
    }
    fn print_combinations(rbs: &[RoaringBitmap]) -> String {
        let mut s = String::new();
        for rb in rbs {
            s.push_str(&format!("{}\n", &display_bitmap(rb)));
        }
        s
    }

    // TODO:
    // - test when a level should be empty
    // - find a way to limit the memory consumption of this ComputeCombinations structure

    #[test]
    fn compute_combinations_4() {
        let b0: RoaringBitmap = (0..).into_iter().map(|x| 2 * x).take_while(|x| *x < 150).collect();
        let b1: RoaringBitmap = (0..).into_iter().map(|x| 3 * x).take_while(|x| *x < 150).collect();
        let b2: RoaringBitmap = (0..).into_iter().map(|x| 5 * x).take_while(|x| *x < 150).collect();
        let b3: RoaringBitmap = (0..).into_iter().map(|x| 7 * x).take_while(|x| *x < 150).collect();

        let parts_candidates = vec![b0, b1, b2, b3];

        let combinations = ComputeCombinations::new(parts_candidates);

        insta::assert_snapshot!(print_compute_combinations(&combinations), @r###"
        Level 1:
            0 [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, ]
            1 [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, ]
            2 [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, ]
            3 [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, ]
        Level 2:
            1 [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, ]
            2 [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, ]
            3 [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, ]
            2 [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, ]
            3 [0, 21, 42, 63, 84, 105, 126, 147, ]
            3 [0, 35, 70, 105, 140, ]
        Level 3:
            2 [0, 30, 60, 90, 120, ]
            3 [0, 42, 84, 126, ]
            3 [0, 70, 140, ]
            3 [0, 105, ]
        Level 4:
            3 [0, ]
        "###);

        let combinations = combinations.finish();
        insta::assert_snapshot!(print_combinations(&combinations), @r###"
        [0, ]
        [30, 42, 60, 70, 84, 90, 105, 120, 126, 140, ]
        [6, 10, 12, 14, 15, 18, 20, 21, 24, 28, 35, 36, 40, 45, 48, 50, 54, 56, 63, 66, 72, 75, 78, 80, 96, 98, 100, 102, 108, 110, 112, 114, 130, 132, 135, 138, 144, 147, ]
        [2, 3, 4, 5, 7, 8, 9, 16, 22, 25, 26, 27, 32, 33, 34, 38, 39, 44, 46, 49, 51, 52, 55, 57, 58, 62, 64, 65, 68, 69, 74, 76, 77, 81, 82, 85, 86, 87, 88, 91, 92, 93, 94, 95, 99, 104, 106, 111, 115, 116, 117, 118, 119, 122, 123, 124, 125, 128, 129, 133, 134, 136, 141, 142, 145, 146, 148, ]
        "###);
    }
    #[test]
    fn compute_combinations_1() {
        let b0: RoaringBitmap = (0..).into_iter().map(|x| 2 * x).take_while(|x| *x < 150).collect();

        let parts_candidates = vec![b0];

        let combinations = ComputeCombinations::new(parts_candidates);

        insta::assert_snapshot!(print_compute_combinations(&combinations), @r###"
        Level 1:
            0 [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, ]
        "###);

        let combinations = combinations.finish();
        insta::assert_snapshot!(print_combinations(&combinations), @r###"
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, ]
        "###);
    }
    #[test]
    fn compute_combinations_2() {
        let b0: RoaringBitmap = (0..).into_iter().map(|x| 2 * x).take_while(|x| *x < 150).collect();
        let b1: RoaringBitmap = (0..).into_iter().map(|x| 3 * x).take_while(|x| *x < 150).collect();

        let parts_candidates = vec![b0, b1];
        let combinations = ComputeCombinations::new(parts_candidates);

        insta::assert_snapshot!(print_compute_combinations(&combinations), @r###"
        Level 1:
            0 [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, ]
            1 [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, ]
        Level 2:
            1 [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, ]
        "###);

        let combinations = combinations.finish();
        insta::assert_snapshot!(print_combinations(&combinations), @r###"
        [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, ]
        [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28, 32, 33, 34, 38, 39, 40, 44, 45, 46, 50, 51, 52, 56, 57, 58, 62, 63, 64, 68, 69, 70, 74, 75, 76, 80, 81, 82, 86, 87, 88, 92, 93, 94, 98, 99, 100, 104, 105, 106, 110, 111, 112, 116, 117, 118, 122, 123, 124, 128, 129, 130, 134, 135, 136, 140, 141, 142, 146, 147, 148, ]
        "###);
    }
}
