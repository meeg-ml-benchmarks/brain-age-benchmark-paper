import shutil
from pathlib import Path
from datetime import datetime
import glob

from mako.template import Template
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


matplotlib.use('Agg')

ROOT = Path(__file__).parent
BUILD_DIR = ROOT / "build"
BUILD_DIR_FIGURES = BUILD_DIR / "figures"

TEMPLATE_INDEX = ROOT / "templates" / "index.mako.html"


def plot_boxplot(df, dataset):
    data = df[df["dataset"] == dataset]

    sns.set_theme(style="ticks")

    f, ax = plt.subplots(figsize=(6, 4))

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="MAE", y="benchmark", data=data,
                whis=[0, 100], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(x="MAE", y="benchmark", data=data,
                  size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    return f


def generate_plots(df):
    """Generate all possible plots for a given benchmark.

    Parameters
    ----------
    df : instance of pandas.DataFrame
        The benchmark results.

    Returns
    -------
    figs : dict
        The matplotlib figures.
    """
    BUILD_DIR_FIGURES.mkdir(exist_ok=True, parents=True)

    dataset_names = df['dataset'].unique()

    figures = {}
    for data_name in dataset_names:
        fig = plot_boxplot(df, data_name)
        fname_short = f"{data_name}"
        figures[data_name] = export_figure(
            fig, f"{fname_short}"
        )
        figures[data_name]["title"] = data_name
    return figures


def export_figure(fig, fig_name):
    if hasattr(fig, 'to_html'):
        return fig.to_html(include_plotlyjs=False)

    fig_basename = f"{fig_name}.svg"
    save_name = BUILD_DIR_FIGURES / fig_basename
    fig.savefig(save_name, bbox_inches='tight')
    plt.close(fig)
    return {"fig_fname": f'figures/{fig_basename}'}


def render_index():
    fnames = sorted(glob.glob(str(ROOT / ".." / "results/*.csv")))
    df = pd.concat(
        [pd.read_csv(f) for f in fnames],
        axis=0
    ).iloc[:, 1:]

    figures = generate_plots(df)

    return Template(filename=str(TEMPLATE_INDEX),
                    input_encoding="utf-8").render(
        figures=figures,
        nb_total_benchs=df["dataset"].nunique(),
        last_updated=datetime.now(),
    )


def copy_static():
    dst = BUILD_DIR / 'static'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / 'static', dst)


def main():

    copy_static()

    rendered = render_index()
    index_filename = BUILD_DIR / 'index.html'
    print(f"Writing index to {index_filename}")
    with open(index_filename, "w") as f:
        f.write(rendered)


if __name__ == "__main__":
    main()
