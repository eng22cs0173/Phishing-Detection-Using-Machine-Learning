import joblib
import re

print("Loading model...")

model = joblib.load(
    r"D:\sneha\smartshield_new\models\html_model.pkl"
)

vectorizer = joblib.load(
    r"D:\sneha\smartshield_new\models\html_vectorizer.pkl"
)

def clean_html(text):

    text = str(text)

    text = re.sub(
        r"<script.*?>.*?</script>",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"<style.*?>.*?</style>",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    return text.lower()


print("Paste HTML content below:")

html_input = input()

cleaned = clean_html(html_input)

vector = vectorizer.transform([cleaned])

prediction = model.predict(vector)[0]

probability = model.predict_proba(vector)[0]

confidence = max(probability)

if prediction == 1:

    result = "PHISHING"

else:

    result = "SAFE"

print("\nResult:", result)

print("Confidence:", confidence)