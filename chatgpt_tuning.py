import sys
# sys.path.append("/Users/pablo/opt/anaconda3/lib/python3.9/site-packages/gradio")

import openai
import gradio

openai.api_key = "sk-"  # INSERT OpenAI key here



messages = [{"role": "system",
            "content": "You are an experienced climate change scientist with expertise in academic literature synthesis, systematic reviews, text mining, and web scraping. Your research focuses\
            on assessing the alignment between media and academic research in their coverage of climate change issues in the context of Chile during the period 2012-2022. Using text mining,\
            web scraping, and topic modeling techniques, you aim to examine the content and compare the thematic focus of climate change discourse in both sources.\
            The objective of your research is to contribute to the understanding of similarities, discrepancies, and gaps in climate change coverage in Chile. By identifying these patterns,\
            you aim to inform future efforts to improve the alignment and comprehensiveness of climate change communication between the media and academia, ultimately promoting public\
            awareness and understanding of this critical global issue."}]


def CustomChatGPT(user_input):

    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(

        model = "gpt-3.5-turbo",

        messages = messages

    )

    ChatGPT_reply = response["choices"][0]["message"]["content"]

    messages.append({"role": "assistant", "content": ChatGPT_reply})

    return ChatGPT_reply



examples = [

    ["Edit this academic paragraph to improve the coherence and flow of writing:"],

    ["Rewrite this paragraph to make it more concise:"],

    ["Could you please provide a brief response to the following comment made by the reviewer of the journal Frontiers in Communication?"],

    ["Using response to reviewer style, edit this paragraph to  improve the coherence and flow of writing: "],

    ["Write a paragraph on []. Suggest a title and scientific references. Use academic style and passive voice."],

    ["As act a author replying to reviewers and please provide a brief and formal response to the following comment made by the reviewer of the journal Frontiers in Communication. Please suggest how to improve the manuscript content:"],

    ["From the following keywords obtained from the Latent Dirichlet Allocation analysis, Suggest a short and informative topic name. The topics fall into five broad categories: climate change indicators, climate change impact, climate change and society, climate policy, and coping with climate change. Classify the suggested name to one of the above topics:"]]







demo = gradio.Interface(fn=CustomChatGPT,

                        inputs=gradio.inputs.Textbox(lines=3, label="What do you need?",placeholder="Text here"),

                        outputs=gradio.outputs.Textbox(label="Habibi, here's my contribution for you:"),

                        title = "Digital Pedro: FA Specialist 💬🤖",

                        examples=examples,

                        show_progress = False)



demo.launch(share=True)