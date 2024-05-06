from tqdm import tqdm

def train_offline(agent, iterations):
    print('Beginning offline training')
    for _ in tqdm(range(iterations)):
        agent.do_eval()

