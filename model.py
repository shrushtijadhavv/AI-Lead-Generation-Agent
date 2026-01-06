import google.generativeai as genai
genai.configure(api_key="AIzaSyBmsLovmHtNEJL_XjQ3yB71VNRmVBfiFlw")
for m in genai.list_models():
    print(m.name)