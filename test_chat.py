import speech_recognition as sr
import pyttsx3
import openai

# 设置你的 API 密钥
openai.api_key = 'YOUR APIKEY'  # 用刚才复制的api key替换单引号里面的内容


# 对话核心
def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        temperature=1,
        n=1,
        stop=None
    )
    reply = response.choices[0].text.strip()
    return reply


# 初始化语音识别器和语音合成器
recognizer = sr.Recognizer()
engine = pyttsx3.init()


def listen():
    with sr.Microphone() as source:
        print("请开始说话...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='zh-CN')
        print("User:", text)
        return text
    except sr.UnknownValueError:
        print("抱歉，无法识别你说的话")
        return "未识别到语音"
    except sr.RequestError:
        print("抱歉，发生了一些错误")

    return ""


# 语音输出
def speak(text):
    print("Chatgpt:", text)
    engine.say(text)
    engine.runAndWait()


# 主程序循环
while True:  # 常驻开机循环
    call_text = listen()  # call_text为唤醒变量
    while "语音助手" in call_text:  # 说“语音助手”，说“退出”之前，会一直循环
        speak("您好，我是您的智能语音助手，现在可以说出您的问题")
        while True:
            input_text = listen()  # input_text为对话时语音输入的变量
            if "退出" in input_text:
                speak("好的，您若有任何需要，请再次呼唤语音助手，再见！")
                call_text = ""
                break
            if "未识别到语音" in input_text:
                speak("抱歉，我无法识别到您的提问")
            else:
                # 根据输入做出相应回答
                # 这里可以根据你的需求添加更多的对话逻辑
                chat_prompt = input_text
                chat_reply = chat_with_gpt(chat_prompt)
                speak(chat_reply)
    if "关机" in call_text and not "确认" in call_text:
        speak("关机之后，再次见到我需要重新运行程序，请您确认是否关机。若要关机请说确认关机")

        call_text = call_text + "未识别到语音"
    if "确认" in call_text:  # 确认是否关机，退出主循环
        speak("好的，再见")
        break
    if not "未识别到语音" in call_text:
        speak("现在默认处于待机模式。若想开启对话，请呼唤语音助手。")