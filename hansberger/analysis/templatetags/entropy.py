from django import template
import matplotlib.pyplot as plt
import mpld3

register = template.Library()


@register.filter
def plot_entropy(obj, normalized):
    entropies = obj.get_entropy_data(normalized)
    plt.figure(figsize=(10, 5))
    for key in entropies:
        plt.plot(entropies[key], '-o')
    plt.legend([key for key in entropies])
    figure = plt.gcf()
    html_figure = mpld3.fig_to_html(figure, template_type='general')
    plt.close()
    return html_figure
