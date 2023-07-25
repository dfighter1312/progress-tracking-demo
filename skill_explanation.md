**Disclaimer**: *This is a conversation-based progress tracking system, and it is just used for experimental purpose. Further details for lifetime-based progress tracking will be discussed on Slack or meeting sessions*

### What is this?

The progress tracking system tries its best to estimate basic ability of a child during their development stage, considering skills that are the most significants one in a rapidly developed world.
Currently, the system analyzes only from chat history using traditional NLP techniques. Advanced techniques should be considered in later stages.

### Which feature is encountered in this version?

All extracted criterion below is on a scale from 0 to 100.

1. **Word uniqueness**

Word uniqueness is the indicator whether a child use many different words in its conversation.

2. **Lexical diversity**

Lexical diversity is one aspect of *lexical richness* and refers to the ratio of different unique word stems (types) to the total number of words (tokens). The term is used in applied linguistics and is quantitatively calculated using numerous different measures. In this context, we are using measure of textual lexical diversity [*McCarthy, Phillip (2005). "An assessment of the range and usefulness of lexical diversity measures and the potential of the measure of textual, lexical diversity (MTLD)". Doctoral Dissertation – via Proquest Dissertations and Theses. (UMI No. 3199485).*]

3. **Readibility difficulty**

The origin of this is *readibility*, which is ease with which a reader can comprehend a body of written text. Readibility can be quantitatively accessed via Reading Ease Score, calculated by [Flesch–Kincaid readability tests](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests). For example, a reading ease score of 90-100 indicates a 5th grade (or less) writing level, while a score of 0-30 indicates a college graduation level. Readibility difficulty is an inverse of readibility, where a high readibility briefly indicates a good writing ability.

4. **Proportion of questions**

Proportion of questions is the number of questions included in the conversation that the child asks. In this version, a naive implementation is implemented, where only sentences with question marks and sentences with asking word in English (who, what, how, etc.) is counted. A score of 100 showing that the child always asks in the conversation.

5. **Proportion of grammatical and spelling error**

This is calculated by the number of writing error divided by the number of words. In the computation, we are taking the subtraction from 100 since higher score is better ability. A score of 100 indicates zero grammatical error.

Note that this grammar check is only available in English, and AI agent's name is not added to the vocabulary, meaning that input any name will be counted as a single error.

6. **Average sentence length**

Number of words comprised in a sentence. Higher number of words per sentences indicates a good word manipulation skill.

7. **Compound sentiment variance**

Variance of compound score in sentiment analysis, whereas compound corresponds to the sum of the valence score of each word in the lexicon and determines the degree of the sentiment rather than the actual value as opposed to the previous ones. Our hypothesis assumes that a child with good emotional intelligence will discover different states of compound.

8. **Average positive rate**

This is the average score of positive feelings that can be inferred from the prompt.

9. **Engagement score**

Engagement is calculated as the similarity between the child's prompt and the last assistant's prompt. It indicates a continuation of conversation within a specific topic. The naive implementation using SBERT deployed in Hugging Face, and accessed via API calls.

10. **Personal connection**

Personal connection can be inferred from a child prompt when he/she mention about him/herself. This is calculated by retrieving personal keyword, such as I, my, mine, or myself, and converted into score by the number of personal keyword divided by the number of sentences.

### Contributed features for each skill

**Creative thinking**

- Word uniqueness
- Lexical diversity
- Readibility difficulty

**Problem solving**

- Proportion of questions

**Communication**

- Proportion of grammar errors
- Average positive rate
- Average sentence length
- Readibility (not difficulty since we encourage simpleness in communication)

**Emotional awareness**

- Compound sentiment variance
- Average sentence length
- Personal connection

**Critical thinking**

- Lexical diversity
- Engagement score

**Time management**: Currently None, but we will consider session time (as the product team suggest) into this skill. Moreover, "time" keywords can be proposed (such as plan, organize, one week, two hours, etc.) to infer time management skill based on keyword and phrase extraction.