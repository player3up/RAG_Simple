# frontend.py
import streamlit as st
import requests
import os

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="智能知识库问答", layout="centered")

st.title("📚 智能知识库问答系统")
st.markdown("上传你的文档（PDF, TXT, DOCX），然后开始提问吧！")

with st.sidebar:
    st.header("1. 上传文档")
    uploaded_file = st.file_uploader(
        "选择一个文件",
        type=["pdf", "txt", "docx"],
        help="支持PDF, TXT, DOCX格式"
    )
    if uploaded_file is not None:
        st.success(f"文件 `{uploaded_file.name}` 已选择")
        if st.button("上传并处理"):
            with st.spinner("正在处理文档，请稍候..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/octet-stream")}
                response = requests.post(f"{BACKEND_URL}/upload_doc", files=files)
                if response.status_code == 200:
                    st.success("✅ 文档处理成功！现在可以开始提问了。")
                else:
                    st.error(f"❌ 处理失败: {response.json().get('detail', '未知错误')}")

st.header("2. 提问")
question = st.text_input("输入你的问题：", placeholder="例如：这份文档的核心观点是什么？")

if st.button("提问"):
    if not question:
        st.warning("请输入问题。")
    else:
        with st.spinner("正在思考..."):
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                result = response.json()
                st.subheader("💡 答案")
                st.write(result["answer"])

                if result["sources"]:
                    st.subheader("📖 参考来源")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(f"来源 {i + 1}: {os.path.basename(source['source'])}"):
                            st.write(source["content"])
            else:
                st.error(f"❌ 请求失败: {response.json().get('detail', '未知错误')}")
