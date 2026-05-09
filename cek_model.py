import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY tidak ditemukan di .env")
    exit()

genai.configure(api_key=api_key)

print("\n✅ Model Gemini yang tersedia untuk generateContent:\n")
print(f"{'No':<4} {'Nama Model':<45} {'Display Name'}")
print("-" * 80)

count = 0
for i, model in enumerate(genai.list_models(), 1):
    if "generateContent" in model.supported_generation_methods:
        count += 1
        print(f"{count:<4} {model.name:<45} {model.display_name}")

print("-" * 80)
print(f"\nTotal: {count} model tersedia")
print("\n👆 Salin salah satu nama model di atas ke app.py\n")
