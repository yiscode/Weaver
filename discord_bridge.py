import os
import discord
from discord.ext import commands
import requests
import json
from dotenv import load_dotenv

load_dotenv()
# --- 設定 ---
TOKEN = os.getenv("API_KEY")
RAG_URL = "http://localhost:9527/world" # 確保與 FastAPI 端口一致

# 設定權限：必須開啟 message_content 才能讀取訊息
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ 黑暗領主 {bot.user} 已上線，正在監控世界線...")
    print(f"正在使用的 RAG 伺服器: {RAG_URL}")

@bot.event
async def on_message(message):
    # 1. 排除 Bot 自己的訊息
    if message.author == bot.user:
        return

    # 2. 判斷觸發條件：被標記 (@機器人) 或 是私訊
    if bot.user in message.mentions or isinstance(message.channel, discord.DMChannel):

        # 使用 clean_content 移除 @機器人 的標籤字串，讓 AI 只讀到純文字
        user_input = message.clean_content.replace(f'@{bot.user.display_name}', '').strip()

        if not user_input:
            await message.reply("🔮 你在黑暗中低語著我的名字，有何吩咐？")
            return

        # 3. 顯示「正在輸入中...」
        async with message.channel.typing():
            # 準備傳送給 RAG Server 的資料
            payload = {
                "query": user_input,
                "user_name": message.author.display_name, # 玩家暱稱，讓 AI 認得你
                "session_id": str(message.channel.id),    # 用 Channel ID 區分不同對話
                "top_k": 5
            }

            try:
                # 4. 呼叫你的 FastAPI (RAG 系統)
                response = requests.post(RAG_URL, json=payload, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    # 取得 AI 產生的敘事內容
                    content = result.get("content", "（空氣中瀰漫著沉默，沒有回應...）")

                    # 5. 回覆玩家
                    await message.reply(content)
                else:
                    await message.channel.send(f"⚠️ 門扉被封印了 (HTTP {response.status_code})")

            except Exception as e:
                await message.channel.send(f"💀 虛空產生了裂縫 (連線錯誤): {e}")

    # 6. 確保其他指令 (如 !reset) 仍能運作
    await bot.process_commands(message)

# --- (選配) 加入一個重置指令 ---
@bot.command()
async def reset(ctx):
    """輸入 !reset 清空該頻道的劇情記憶"""
    payload = {
        "query": "reset request", # 雖然 reset 不需要 query，但配合 Data Model 還是帶一下
        "user_name": ctx.author.display_name,
        "session_id": str(ctx.channel.id)
    }

    try:
        # 注意這裡 URL 改成 /reset
        response = requests.post(f"http://localhost:9527/reset?session_id={ctx.channel.id}")
        if response.status_code == 200:
            await ctx.send("✨ 命運的絲線已被重整，這個頻道的世界已回歸最初狀態。")
        else:
            await ctx.send("❌ 儀式失敗，無法抹除記憶。")
    except Exception as e:
        await ctx.send(f"💀 系統錯誤: {e}")

bot.run(TOKEN)