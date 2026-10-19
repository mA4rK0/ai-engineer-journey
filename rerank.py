from sentence_transformers import CrossEncoder

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(query, docs, top_n=3):
    model = get_model()
    pairs = [(query, d) for d in docs] 
    scores = model.predict(pairs)        
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    top = ranked[:top_n]
    return top, [float(scores[i]) for i in top]

    