import json

d = json.load(open('eval/eval_results.json'))
for r in d['results']:
    eid = r['eval_id']
    s = r['scores']
    total = s['factual_accuracy'] + s['citation_quality'] + s['reasoning_trace'] + s['completeness']
    if total < 10:
        print(f"\n=== {eid} ({r['difficulty']}) Total: {total}/10 FA:{s['factual_accuracy']} CQ:{s['citation_quality']} RT:{s['reasoning_trace']} CO:{s['completeness']} ===")
        print(f"Q: {r['question'][:120]}")
        print(f"CoT: {r['chain_of_thought'][:300]}")
        print(f"Answer: {r['ai_answer'][:400]}")
        print(f"Judge: {s['reasoning']}")
        print(f"Cited: {[c['doc_id'] for c in r['citations']]}")
        print(f"Expected: {r['source_doc_ids_expected']}")
