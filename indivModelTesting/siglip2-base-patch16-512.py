from transformers import pipeline

# load pipeline
ckpt = ["google/siglip2-base-patch16-512", "google/siglip2-base-patch16-384"]
image_classifier = pipeline(model=ckpt[1], task="zero-shot-image-classification")

# load image and candidate labels
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
candidate_labels = ["2 cats", "a plane", "a remote"]

# run inference
outputs = image_classifier(url, candidate_labels=candidate_labels)
print(outputs)
