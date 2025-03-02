import pandas as pd
import language_tool_python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load and preprocess the data 
df = pd.read_csv("yelp_review.csv")
df1 = pd.read_csv("yelp_business.csv")
df3 = pd.merge(df1, df, on='business_id', how="outer")
df3.drop(
    columns=["neighborhood", "latitude", "longitude", "is_open", "review_id", "user_id", "date", "useful", "funny", "cool", "review_count", "stars_y"],
    inplace=True
)
df4 = df3[df3["state"] == "PA"][['name', 'text']]
df4 = df4.groupby('name').head(5)
df5 = df4.groupby('name')['text'].apply(lambda rows: ' + '.join(rows)).reset_index()
for i in range(len(df5)):
    df5.loc[i, 'name'] = df5.loc[i, 'name'].strip('"')

# Load model, tokenizer, and language tool only once
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
lang_tool = language_tool_python.LanguageTool('en-US')

def get_business_reviews(business_name):
    reviews = df5[df5['name'] == business_name]['text'].tolist()
    return " ".join(reviews)

def generate_summary(text, bullet_points):
    prompt = (
        "summarize: Summarize the following reviews into 3 distinct bullet points that are grammatically correct and provide clear context.\n\n"
        "Reviews:\n" + text
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(
        input_ids,
        max_length=500,
        min_length=50,
        num_beams=6,
        early_stopping=True
    )
    
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    sentences = [s.strip() for s in summary.split('. ') if s.strip()]
    bullet = ["- " + sen for sen in sentences][:bullet_points]
    
    return bullet

def correct_grammar(bullet_list):
    points = []
    for point in bullet_list:
        corrected = lang_tool.correct(point)
        points.append(corrected)
    return points

def get_prediction(business_name):
    reviews = get_business_reviews(business_name)
    summary = generate_summary(reviews, bullet_points=3)
    corrected_summary = correct_grammar(summary)
    return corrected_summary

# only for local testing.  
if __name__ == "__main__":
    business_name = "Eazor's Auto Salon"
    final_summary = get_prediction(business_name)
    for bullet in final_summary:
        print(bullet)
