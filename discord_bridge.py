import discord
import os
from dotenv import load_dotenv
import httpx  # 務必先 pip install httpx
import asyncio
from discord.ext import commands
load_dotenv()
# --- 設定 ---
# 設定參數
TOKEN = os.getenv("API_KEY")
RAG_URL = "http://localhost:9527"  # 指向你的 FastAPI 伺服器

intents = discord.Intents.default()
intents.message_content = True  # 務必在 Discord Developer Portal 開啟此權限
bot = commands.Bot(command_prefix="!", intents=intents)

# 建立全局非同步 Client
async_client = httpx.AsyncClient(
    timeout=300.0,
)

@bot.event
async def on_ready():
    print(f"✅ Weaver Bridge 已連線: {bot.user}")
    print(f"👉 模式：僅回應標記 (@{bot.user.name}) 與 !reset 指令")

# --- 功能 1：Reset 重置指令 ---
@bot.command(name="reset")
async def reset(ctx):
    """重置該頻道的劇情與記憶"""
    session_id = str(ctx.channel.id)
    try:
        # 呼叫 FastAPI 的 /reset 接口
        response = await async_client.post(f"{RAG_URL}/reset", params={"session_id": session_id})
        if response.status_code == 200:
            await ctx.send(f"🧹 頻道 {session_id} 的記憶已完全抹除。管理員已介入。")
        else:
            await ctx.send("❌ 重置失敗，請檢查 RAG 伺服器狀態。")
    except Exception as e:
        await ctx.send(f"❌ 重置時發生錯誤: {e}")

# --- 功能 2：訊息處理 (Tag 觸發) ---
@bot.event
async def on_message(message):
    # 排除機器人自己的訊息
    if message.author == bot.user:
        return

    # 優先處理指令 (如 !reset)
    await bot.process_commands(message)

    # 判斷是否標記了機器人
    if bot.user.mentioned_in(message):
        # 移除訊息中的標記標籤，只留下純文字 query
        clean_content = message.content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()

        if not clean_content:
            await message.channel.send("（機器人正冷冷地看著你，等待你的指令...）")
            return

        async with message.channel.typing():
            payload = {
                "query": clean_content,
                "user_name": message.author.name,
                "player_id": str(message.author.id),
                "session_id": str(message.channel.id)
            }

            try:
                # 呼叫 FastAPI 的 /world 接口
                response = await async_client.post(f"{RAG_URL}/world", json=payload)
                response.raise_for_status()

                data = response.json()
                content = data.get("content", "系統無回應")

                # 分段傳送長訊息
                if len(content) > 2000:
                    for i in range(0, len(content), 2000):
                        await message.channel.send(content[i:i+2000])
                else:
                    await message.channel.send(content)

            except httpx.ReadTimeout:
                await message.channel.send("⚠️ [警告]：中控系統響應超時，異想體能量波動過大。")
            except Exception as e:
                print(f"❌ 錯誤: {e}")
                await message.channel.send(f"❌ 系統異常: {str(e)}")

# 關閉時安全釋放資源
@bot.event
async def on_close():
    await async_client.aclose()

bot.run(TOKEN)