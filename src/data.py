import json
from torch.utils.data import Dataset


class SQuAD(Dataset):
    def __init__(self, path):
        super().__init__()
        contexts, questions = self._load_squad(path)
        self.contexts = contexts
        self.questions = questions
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question = self.questions[index]['question']
        context_id = self.questions[index]['context_id']
        context = self.contexts[context_id]
        return question, context
    
    def _load_squad(self, path):
        # Load json file
        with open(path, 'r') as f:
            squad = json.loads(f.read())
        
        # Parse json
        contexts = []
        questions = []
        ctx_id = 0
        for topic in squad['data']:
            for paragraph in topic['paragraphs']:
                contexts.append(paragraph['context'])
                questions += [{'question': qa['question'],
                            'context_id': ctx_id, 
                            'answers': qa['answers']}
                            for qa in paragraph['qas']]
                ctx_id += 1
        
        return contexts, questions