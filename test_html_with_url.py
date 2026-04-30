import joblib
import requests
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


# 👇 Put test URLs here
urls = [

"https://www.google.com",
"https://www.youtube.com",
"https://www.github.com"

]

for url in urls:

    print("\nTesting:", url)

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        html = response.text

        cleaned = clean_html(html)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = max(probability)

        if prediction == 1:

            result = "PHISHING"

        else:

            result = "SAFE"

        print("Result:", result)
        print("Confidence:", confidence)

    except Exception as e:

        print("Failed to fetch:", url)