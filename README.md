# LLM Chatbot with Private Dataset

ระบบ Chatbot ที่ใช้ **RAG** ดึงข้อมูลจากฐานข้อมูลกระทู้ Agnos Health Forum  
แล้วใช้ LLM (เช่น **LLaMA 3** ผ่าน Ollama) เพื่อสร้างคำตอบที่ถูกต้องและอ้างอิงได้

## Overview

โครงการนี้เป็นการสร้าง LLM Chatbot ที่เชื่อมต่อกับ Private Dataset จาก Agnos Health Forum โดยใช้เทคนิค RAG เพื่อให้ Chatbot สามารถค้นหาและตอบคำถามจากข้อมูลในฐานข้อมูลได้อย่างถูกต้องและครบถ้วน

ระบบใช้ LLM (LLaMA 3 via Ollama) สำหรับประมวลผลภาษา และ Vector Database สำหรับค้นหาข้อมูลที่ใกล้เคียงที่สุดจากเอกสาร

## Project Structure

- Collection&Preprocessing.py: รวบรวมและทำความสะอาดข้อมูล
- Embedding&Indexing.py: สร้าง embeddings และ index
- Search_QA.py: Chat interface (Streamlit + LLaMA)
- vector_index: เก็บไฟล์ vector database
- agnos_forum_posts.json: Dataset
- requirements.txt
- README.md

## Clone Repo

```bash
git clone https://github.com/fluke-suphanut/LLM-Chatbot-with-Private-Dataset.git
cd LLM-Chatbot-with-Private-Dataset
```

## สร้าง Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

## ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

## ติดตั้ง Ollama และโหลดโมเดล LLaMA

ดาวน์โหลด: https://ollama.ai/download
โหลดโมเดล LLaMA 3 ขนาด 8B

```bash
ollama pull llama3:8b
```

## รัน chatbot ด้วย Streamlit

```bash
streamlit run Search_QA.py
```

## Features

- RAG Search: ค้นหาข้อมูลที่เกี่ยวข้องจากฐานข้อมูล forum
- LLM Response: สร้างคำตอบโดยใช้ LLaMA 3
- Multi-turn Chat: จำประวัติการสนทนา
- Reference Links: แสดงเอกสารอ้างอิง
- Website Scraper : ดึงข้อมูลอัตโนมัติจาก Agnos Health Forum เพื่ออัปเดตฐานข้อมูล

## Model Pipeline Construction

- User Input: ผู้ใช้พิมพ์คำถามใน UI
- Retriever: ค้นหาเอกสารที่เกี่ยวข้องจาก Vector DB
- LLM Generation: ให้ LLaMA 3 ประมวลผลและสร้างคำตอบ
- Response + References: ส่งคำตอบกลับไปยังผู้ใช้พร้อมแหล่งข้อมูล

## output

[here](output.pdf)

