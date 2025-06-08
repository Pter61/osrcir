from cloudgpt_aoai import get_chat_completion, encode_image
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import time
import random

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(60))
def openai_completion_vision_CoT(sys_prompt, user_prompt, image, engine="gpt-4-turbo-20240409", max_tokens=1024, temperature=0):
    global_attempt, local_attempt = 0, 0
    global_max_attempts, local_max_attempts = 2, 3
    # make multiple attempts to handle occasional failures
    while global_attempt < global_max_attempts:
        try:
            try:
                return attempt_openai_completion_CoT(sys_prompt, user_prompt, image, engine, max_tokens, temperature)
            except Exception as e:
                local_attempt += 1
                if local_attempt < local_max_attempts:
                    wait_time = random.randint(1, 60)
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"Error after multiple attempts with engine {engine}: {str(e)} with sample {image}")
                    print("Switching to gpt-4o-20240806 due to multiple failures.")
                    return attempt_openai_completion_CoT(sys_prompt, user_prompt,  image, "gpt-4o-20240806")
        except Exception as e:
            global_attempt += 1
            if global_attempt == global_max_attempts:
                print(f"Bad Error after multiple attempts with engine {engine}: {str(e)} with sample {image}, return None!")
                return ""

def attempt_openai_completion_CoT(sys_prompt, user_prompt, image, engine="gpt-4o-20240806", max_tokens=4096, temperature=0):
    image_url = encode_image(image)
    chat_message = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    resp = get_chat_completion(
        engine=engine,
        messages=chat_message,
        max_tokens=max_tokens,
        timeout=10,
        # request_timeout=10,
        temperature=temperature,
        stop=["\n\n", "<|endoftext|>"]
    )
    # print(resp.choices[0].message)
    # print(resp)
    print("\n%s: %s" % (engine, resp.choices[0].message.content))
    return resp.choices[0].message.content
