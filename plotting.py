import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.style.use('dark_background')
    plt.clf()
    plt.title('Training Session Progress')
    plt.xlabel('# of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='Green')
    plt.plot(mean_scores, color = 'DarkGreen')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))