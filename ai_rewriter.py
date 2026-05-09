"""
DeepSeek API 调用模块
兼容 OpenAI 接口格式
"""

import streamlit as st
from openai import OpenAI


def get_deepseek_client():
    """获取 DeepSeek API 客户端"""
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        # 如果 secrets 中没有，尝试从环境变量获取
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def rewrite_text(text: str, mode: str = "balanced", intensity: str = "medium") -> str:
    """
    调用 DeepSeek API 改写文本
    
    Args:
        text: 要改写的原始文本
        mode: 改写模式 ('reduce_ai', 'reduce_plagiarism', 'balanced')
        intensity: 改写强度 ('light', 'medium', 'heavy')
    
    Returns:
        改写后的文本
    """
    from utils.prompts import get_prompt
    
    if not text or len(text.strip()) == 0:
        return "请输入需要改写的文本内容。"
    
    # 获取对应的 Prompt
    system_prompt = get_prompt(mode, intensity)
    
    try:
        client = get_deepseek_client()
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=4096,
            stream=False
        )
        
        result = response.choices[0].message.content
        
        # 清理可能的引号包裹
        result = result.strip()
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return "❌ API Key 无效，请检查配置。"
        elif "429" in error_msg or "rate" in error_msg.lower():
            return "❌ 请求过于频繁，请稍后再试。"
        elif "timeout" in error_msg.lower():
            return "❌ 请求超时，请检查网络连接后重试。"
        else:
            return f"❌ 改写失败：{error_msg}"
