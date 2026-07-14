import json

with open('frontend/public/data/text_models.json') as f:
    data = json.load(f)

for rep in ['SBERT', 'TF-IDF']:
    model = data[rep]
    neurons = model['neurons']
    all_docs = set()
    for n in neurons:
        all_docs.update(n['doc_indices'])
    
    class_counts = {}
    for n in neurons:
        dc = n['dominant_class']
        class_counts[dc] = class_counts.get(dc, 0) + n['total_samples']
    
    print(f"\n=== {rep} ===")
    print(f"Total unique docs in model: {len(all_docs)}")
    print(f"Doc id range: {min(all_docs) if all_docs else 'N/A'} - {max(all_docs) if all_docs else 'N/A'}")
    print(f"Class distribution by samples: {class_counts}")
    print(f"Grid: {model['cols']}x{model['rows']}")

with open('frontend/public/data/news_samples.json') as f:
    samples = json.load(f)
print(f"\n=== news_samples.json ===")
print(f"Total sample entries: {len(samples)}")
for s in samples:
    print(f"  id={s['id']} class={s['class']} text_len={len(s['text'])}")

with open('frontend/public/data/text_metrics.json') as f:
    metrics = json.load(f)
print(f"\n=== text_metrics.json ===")
print(json.dumps(metrics, indent=2))
