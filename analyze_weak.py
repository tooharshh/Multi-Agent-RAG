import json

with open("eval/eval_results.json") as f:
    results = json.load(f)
with open("eval_set_ai_healthcare.json") as f:
    eval_set = json.load(f)

expected = {q["eval_id"]: q for q in eval_set["questions"]}

# Handle both list and dict formats
if isinstance(results, dict):
    result_list = results.get("results", [])
else:
    result_list = results

weak = [r for r in result_list if r["scores"]["total"] < 10]
print(f"Total weak questions: {len(weak)}\n")

for r in weak:
    eid = r["eval_id"]
    exp = expected[eid]
    s = r["scores"]
    print(f"=== {eid} ({r['difficulty']}) Score: {s['total']}/10 FA:{s['factual_accuracy']} CQ:{s['citation_quality']} RT:{s['reasoning_trace']} CO:{s['completeness']} ===")
    print(f"Q: {exp['question'][:150]}")
    print(f"Expected source docs: {exp['source_docs']}")
    retrieved = r.get("source_doc_ids_retrieved", [])
    cited = [c["doc_id"] for c in r.get("citations", [])]
    print(f"Retrieved: {retrieved}")
    print(f"Cited: {cited}")
    # Check overlap
    exp_set = set(exp["source_docs"])
    cited_set = set(cited)
    retrieved_set = set(retrieved)
    print(f"Expected docs in retrieved: {exp_set & retrieved_set}")
    print(f"Expected docs NOT retrieved: {exp_set - retrieved_set}")
    print(f"Expected docs cited: {exp_set & cited_set}")
    print(f"Expected docs NOT cited: {exp_set - cited_set}")
    print(f"Judge: {s.get('reasoning', '')[:300]}")
    print(f"Expected answer (first 300): {exp['expected_answer'][:300]}")
    print(f"AI answer (first 300): {r['ai_answer'][:300]}")
    print()
