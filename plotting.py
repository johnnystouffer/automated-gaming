import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    plt.clf()
    plt.title('Training In Process...')
    plt.xlabel('# of Games')
    plt.ylabel('SCORE')
    plt.plot(scores, color='powderblue')
    plt.plot(mean_scores, color = 'lightpink')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)