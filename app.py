import os
import re
import streamlit as st
import google.generativeai as genai_legacy
from google import genai
from pinecone import Pinecone

# --- 1. UI Setup (חייב להיות ראשון!) ---
st.set_page_config(page_title="OMFS Department AI", page_icon="🦷", layout="centered")

# --- 2. API Setup ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# הגדרת המודלים (הישן ליצירת הטקסט, והחדש לחיפוש הווקטורי)
genai_legacy.configure(api_key=GOOGLE_API_KEY)
llm_model = genai_legacy.GenerativeModel('gemini-3.5-flash')
flash_model = genai_legacy.GenerativeModel('gemini-3.5-flash')
google_client = genai.Client(api_key=GOOGLE_API_KEY)

# --- 3. Database Connection ---
@st.cache_resource
def load_database():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("omfs-kb")
    return index

index = load_database()

# --- 4. UI Design ---
st.title("🦷 OMFS Department Knowledge Base")
st.markdown("Ask any medical question. The AI will provide answers directly from our Cloud Knowledge Base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. User Interaction ---
if user_query := st.chat_input("Enter your medical question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # --- 6. Processing ---
    with st.chat_message("assistant"):
        with st.status("🔍 Processing request...", expanded=True) as status:
            
            st.write("🧠 Optimizing search with medical synonyms...")
            expansion_prompt = f"""
            You are an expert OMFS surgeon. The user wants to search a medical textbook with this query: "{user_query}"
            1. Translate the query to professional medical English.
            2. Expand it with exact anatomical and clinical synonyms.
            3. Return ONLY the expanded search string, nothing else.
            """
            expanded_query = flash_model.generate_content(expansion_prompt).text.strip()
            
            st.write(f"📚 Searching Pinecone cloud database for: {expanded_query}")
            
            # --- השינוי הקריטי: חיפוש באמצעות המודל החדש! ---
            response = google_client.models.embed_content(
                model="gemini-embedding-001",
                contents=expanded_query
            )
            query_vector = response.embeddings[0].values
            
            search_response = index.query(
                vector=query_vector,
                top_k=100,
                include_metadata=True
            )
            
            context_parts = []
            
            # ❗ שים כאן את ה-ID האמיתי של התיקייה בדרייב שלך ❗
            FOLDER_ID = "1fGrnIlCfLqXxnZF-EsFKmFNq6SLeDPri" 
            
            for match in search_response['matches']:
                metadata = match['metadata']
                text = metadata.get('text', '')
                source = metadata.get('source', 'Unknown Source')
                page_num = metadata.get('page', 0)
                
                title_match = re.search(r"title:\s*(.+)", text)
                smart_title = title_match.group(1).strip() if title_match else source
                
                drive_link = f"https://drive.google.com/drive/u/0/search?q=parent:{FOLDER_ID}%20and%20name:'{source}'"

                context_parts.append(f"--- SOURCE: [{smart_title}]({drive_link}) (Internal Page {int(page_num) + 1}) ---\n{text}")
            
            context = "\n\n".join(context_parts)
            
            st.write("🧠 Formulating final response...")
            
            prompt = f"""
            You are an expert AI assistant for an Oral and Maxillofacial Surgery (OMFS) department.
            Answer the user's question STRICTLY based on the provided context.
            
            Strict Rules:
            1. Language: ENGLISH.
            2. Depth: Provide an in-depth, professional explanation.
            3. CITATION: Cite your sources at the end using ONLY the document name and internal page provided in the 'SOURCE:' tag.
            
            Context:
            {context}

            Original User Question: {user_query}
            """
            
            final_response = llm_model.generate_content(prompt)
            status.update(label="✅ Answer ready", state="complete", expanded=False)
        
        st.markdown(final_response.text)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_response.text
        })
