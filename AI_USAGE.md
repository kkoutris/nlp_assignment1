# AI assistance record

Tool: Claude Code

## Project 1

### Implementation support from the agent

The agent mainly helped with coding.

| Area | What it did |
|---|---|
| Plan | Gave some input on the experiment plan, which the team designed and finalised |
| Code review | Read the tutorial modules and pointed out issues (early stopping on the scored fold, the `runs` argument, `TokenIndicator.__len__`) |
| Code | Wrote the project code to the team's plan, first as separate scripts, now inline in `imdb_sentiment.ipynb` |
| Figures | Produced the analysis figures |

## Project 2

The agent was asked to solve Project 2 in the same way as Project 1, and did most of the work.

| Area | What it did |
|---|---|
| Plan | Gave some input on the experiment plan, which the team designed and finalised |
| Code | Wrote most project code, inline in `imdb_sentiment_part2.ipynb`, including the faster encoders and the checks that they match the tutorial's |


The notebook doesn't use `part2/project2.py`. A few of its ideas were reimplemented inline: keeping negation and
contrast words when filtering stopwords, and reading only the IMDb words of each embedding table.
