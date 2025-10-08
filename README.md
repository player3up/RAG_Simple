📚 RAG-Powered Knowledge Base Q&A System
一个基于检索增强生成（RAG）技术的智能知识库问答系统，允许用户上传文档（PDF, TXT, DOCX），并通过自然语言提问，从文档中获取准确、有据可查的答案。

 `Python`
 `FastAPI `
 `Streamlit`
 `License: MIT`

✨ 功能特性
📄 多格式文档支持：支持上传和处理 PDF、TXT、DOCX 格式的文档。
🔍 智能检索：基于先进的向量检索技术，精准定位文档中的相关信息。
🤖 多模型支持：灵活切换大语言模型后端，支持本地 Ollama、免费的 SiliconFlow API 或 OpenAI API。
💬 自然语言交互：通过简洁的 Web 界面，用自然语言进行提问。
📖 答案溯源：每个答案都附上其在原文中的出处，确保信息可追溯、可验证。
🌐 RESTful API：提供完整的后端 API，方便与其他应用集成。
🛠️ 技术栈
后端
前端
数据库
LLM 后端
FastAPI	Streamlit	ChromaDB	Ollama / SiliconFlow / OpenAI
LangChain		HuggingFace	

🚀 快速开始
先决条件
Python 3.10+
Miniconda (推荐用于环境管理)
Git
1. 克隆项目
```
git clone https://github.com/your-username/my_rag_project.git
cd my_rag_project
```
2. 创建并激活 Conda 环境
我们提供了一个 environment.yml 文件来轻松配置所有依赖。
```
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate rag_env
```
3. 配置环境变量
复制环境变量示例文件：
```
cp .env.example .env
编辑 .env 文件，根据你选择的 LLM 后端填入相应的 API Key。
使用 SiliconFlow: 填入 SILICONFLOW_API_KEY
使用 OpenAI: 填入 OPENAI_API_KEY
使用 Ollama: 无需填写任何 API Key
```
4. 下载本地模型 (如果使用 Ollama)
如果你选择使用 Ollama，请先下载一个模型。


# 在一个新的终端中运行
`ollama pull qwen:7b`
5. 运行应用
你需要打开两个终端来分别运行后端和前端。

终端 1: 启动后端服务
```
# 确保在 (rag_env) 环境中
conda activate rag_env
python main.py
终端 2: 启动前端应用


# 确保在 (rag_env) 环境中
conda activate rag_env
streamlit run frontend.py
```
应用将在你的浏览器中自动打开 http://localhost:8501。

📖 使用方法
上传文档：在侧边栏点击 "Browse files"，选择一个 PDF、TXT 或 DOCX 文件，然后点击 "上传并处理"。
等待处理：系统会自动解析文档、构建知识库，这个过程可能需要几十秒。
开始提问：在主界面的输入框中输入你的问题，例如 "这份文档的核心观点是什么？"。
查看答案：点击 "提问" 按钮，稍等片刻即可看到生成的答案和参考来源。

📁 项目结构
``` 
my_rag_project/
├── data/                   # 存放上传的文档
├── vector_db/              # ChromaDB 存储向量数据的目录
├── .env.example            # 环境变量示例文件
├── environment.yml         # Conda环境配置文件
├── requirements.txt        # 项目依赖 (备用)
├── rag_core.py            # RAG核心逻辑（加载、分块、检索、生成）
├── main.py                # FastAPI后端服务器
└── frontend.py            # Streamlit前端应用
```
⚙️ 配置
你可以在 rag_core.py 文件中修改以下配置：

EMBEDDING_MODEL: 选择嵌入模型 ("openai" 或 "huggingface")。
LLM_MODEL: 选择大语言模型后端 ("siliconflow", "openai" 或 "ollama")。
SILICONFLOW_MODEL_NAME: 当使用 SiliconFlow 时，指定模型名称。
OLLAMA_MODEL_NAME: 当使用 Ollama 时，指定模型名称。
