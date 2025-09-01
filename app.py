import torch
import streamlit as st
from transformers import T5Tokenizer, AutoModelForCausalLM

# --- モデルロード ---
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("output_small/", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("output_small/", local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- UI 部分 ---
st.title("ボイラQAチャット（GPT-2ベース）")
st.success("モデルのロード完了 ✅")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 既存のチャット履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("質問を入力してください"):
    # ユーザーの質問を履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # モデルからの応答生成
    with st.chat_message("assistant"):
        with st.spinner("推論中..."):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=100,
                    num_return_sequences=1
                )
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        st.markdown(decoded_output)
    # 応答を履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": decoded_output})
