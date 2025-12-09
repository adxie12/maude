from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from queue import Queue
from dataclasses import dataclass
import time
import torch
import threading

#Launch a daemon thread for a specific function
#return that thread handle
def launch(targ, args):
    t = threading.Thread(target=targ, args=args)
    t.daemon = True #make thread die with this program
    t.start()
    return t

def processQLoop(owner, timing = 0.05):
    while True:
        time.sleep(timing)
        owner.processQueue()

@dataclass
class LLMServiceConfig:
    model_name = "Qwen/Qwen3-4B"
    q_refresh_t = 0.25

class LLMService:
    def __init__(self, config):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.inputQ = Queue()
        self.process_thread = launch(processQLoop,(self, config.q_refresh_t))

    # check if a thread is still alive    
    def threadAlive(self, thr):
        if thr == None:
            return False
        thr.join(timeout=0.0)
        return thr.is_alive()

    def checkServiceHealth(self):
        if not self.threadAlive(self.process_thread):
            self.process_thread = launch(processQLoop,(self, self.config.q_refresh_t))

    def submit(self, data, outQ):
        self.inputQ.put((data,outQ))
        # check
        self.checkServiceHealth()

    def processQueue(self):
        while not self.inputQ.empty():
            messages,outQ = self.inputQ.get()

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking = False
            )
            model_inputs = self.tokenizer([text], return_tensors = "pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens = 32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            outQ.put(content)


 