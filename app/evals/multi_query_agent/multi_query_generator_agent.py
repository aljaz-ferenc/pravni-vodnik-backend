from app.agents.multi_query_generator_agent import generate_multi_queries
from app.evals.multi_query_agent.test_queries import test_queries
from app.utils.dedupe_queries import dedupe_queries

TEST_QUERY_INDEX = 2
test_query = test_queries[TEST_QUERY_INDEX]

generated_queries = generate_multi_queries(test_query["query"])
deduped_queries = dedupe_queries(
    generated_queries,
    threshold=test_query["threshold"],
    max_queries=test_query["max_queries"],
)

print(deduped_queries)
