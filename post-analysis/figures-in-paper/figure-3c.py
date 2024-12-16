#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import seaborn as sns
from ipywidgets import fixed, interact, interact_manual, interactive
from natsort import index_natsorted
from scipy.stats import norm


df = pd.read_csv("../../data/3_analysis-resistance/all_rgiout_2022-12-01", sep="\t")
df = df.sort_values(
    by=["Accession_Number"],
    ascending=True,
    key=lambda x: np.argsort(index_natsorted(df["Accession_Number"])),
    ignore_index=True,
)
df[["Species", "Subspecies"]] = df["Strain"].str.split("subsp.", 1, expand=True)
df = df.replace(np.nan, "Null", regex=True)

rgiintegron = pd.read_csv(
    "../../data/3_analysis-resistance/rgi_subset_integronoverlap",
    sep="\t",
    names=df.columns.values,
)
rgiintegron["Species"] = rgiintegron["Strain"].str.split("subsp.", 1, expand=True)
rgiintegron["Subspecies"] = "Null"
contigdf = pd.concat(
    [
        df["Accession_Number"],
        df["Contig"].str.split("_", expand=True)[0].str.split("|").str[-1],
    ],
    axis=1,
).copy()

ecoli_mlplasmid_out = pd.read_csv(
    "../../data/3_analysis-plasmid/merged-Escherichia_coli-mlplasmidout", sep="\t"
)
ecoli_mlplasmid_out["Contig"] = ecoli_mlplasmid_out["Contig_name"].str.split(
    " ", expand=True
)[0]
ecoliplasmids_df = contigdf.loc[
    contigdf["Accession_Number"].isin(ecoli_mlplasmid_out["AccNum"])
]
ecoliplasmids_df = ecoliplasmids_df.loc[
    ecoliplasmids_df[0].isin(ecoli_mlplasmid_out["Contig"])
]
ecoliplasmids_df = df.loc[ecoliplasmids_df.index].copy()


# # Function definitions

def measure_obs_distance(
    dataframe, value, anthro_year, column, sums=False, verbose=False
):
    num_yearcultured_allstrains_dict = (
        dataframe.drop_duplicates(subset=["Accession_Number"])["Year_Cultured"]
        .value_counts()
        .to_dict()
    )
    num_yearcultured_valposstrains_dict = (
        dataframe.loc[dataframe[column].str.contains(value, na=False, regex=True)]
        .drop_duplicates(subset=["Accession_Number"])["Year_Cultured"]
        .value_counts()
        .to_dict()
    )
    fractional_dictionary = {}
    for years in num_yearcultured_allstrains_dict:
        if num_yearcultured_allstrains_dict[years] == 0:
            continue
        if years in num_yearcultured_valposstrains_dict:
            val = [
                num_yearcultured_valposstrains_dict[years],
                num_yearcultured_allstrains_dict[years],
            ]
            # fraction = (
            #     num_yearcultured_valposstrains_dict[years]
            #     / num_yearcultured_allstrains_dict[years]
            # )
        else:
            val = [0, num_yearcultured_allstrains_dict[years]]
        fractional_dictionary[years] = val

    yeardf = (
        pd.DataFrame.from_dict(
            fractional_dictionary, orient="index", columns=["num_pos", "all"]
        )
        .reset_index()
        .rename(columns={"index": "year"})
        .sort_values(by="year")
        .reset_index(drop=True)
    )
    yeardf["frac"] = yeardf["num_pos"] / yeardf["all"]
    if verbose:
        print(yeardf)
    anthro = {True: "Pre-Human", False: "Post-Human"}
    # line = pd.Index(yeardf["year"]).get_loc(anthro_year)
    # yeardf["row"] = np.arange(yeardf.shape[0])
    # yeardf["Anthropogenicity"] = "Pre-Human"
    # yeardf.loc[yeardf["row"] > line, "Anthropogenicity"] = "Post-Human"
    yeardf["Anthropogenicity"] = "Pre-Human"
    yeardf.loc[yeardf["year"] > anthro_year, "Anthropogenicity"] = "Post-Human"

    preanthro_mean = yeardf.loc[yeardf["Anthropogenicity"] == "Pre-Human"][
        "frac"
    ].mean()
    postanthro_mean = yeardf.loc[yeardf["Anthropogenicity"] == "Post-Human"][
        "frac"
    ].mean()
    metric = postanthro_mean - preanthro_mean
    # print("Pre-Human mean fraction = %s" % (preanthro_mean))
    # print("Post-Human mean fraction = %s" % (postanthro_mean))
    # print("metric = %s" % (metric))
    return yeardf, preanthro_mean, postanthro_mean


# shuffle year_cultured information while retaining existing structure.
#  i.e. all 1940 strains get remapped to 2019, all 2019 strains get remapped to 1982, etc.
def shuffleyears_structured(
    dataframe, value, anthro_year, column, verbose=False, simulations=500
):
    null_distances = []
    sortedyears = dataframe["Year_Cultured"].unique()
    for sim in range(simulations):
        copy_df = dataframe.copy()
        shuffledyears = dataframe["Year_Cultured"].sample(frac=1).unique()
        remapping = dict(zip(sortedyears, shuffledyears))
        copy_df["Year_Cultured"] = dataframe["Year_Cultured"].map(remapping)
        yeardf, pre, post = measure_obs_distance(
            copy_df, value, anthro_year, column, verbose
        )
        null_distances.append(post - pre)
    return null_distances


# shuffle year_cultured information while NOT retaining existing structure.
#  i.e. some 1940 strains can get remapped to 2019, some can get remapped to 1930, etc.
def shuffleyears_unstructured(
    dataframe, value, anthro_year, column, verbose=False, simulations=500
):
    null_distances = []
    sortedyears = dataframe["Year_Cultured"].unique()
    uq_strains = (
        dataframe.groupby(["Accession_Number", "Year_Cultured"], sort=False)
        .size()
        .reset_index()
    )
    uq_strains.set_index("Accession_Number", inplace=True)
    uq_strains.drop(columns=[0, "Year_Cultured"], inplace=True)
    for sim in range(simulations):
        uq_strains["RandomChoice"] = np.random.choice(sortedyears, uq_strains.shape[0])
        copy_df = dataframe.copy()
        remapping = uq_strains.to_dict()["RandomChoice"]
        copy_df["Year_Cultured"] = dataframe["Accession_Number"].map(remapping)
        yeardf, pre, post = measure_obs_distance(
            copy_df, value, anthro_year, column, verbose
        )
        null_distances.append(post - pre)
    return null_distances

import scipy.stats as stats
from scipy.stats import binom

def calculate_frequency_bounds(row, confidence=0.95):
    successes = row["num_pos"]
    attempts = row["all"]
    freq = row["num_pos"] / row["all"]
    alpha = successes + 1
    beta = attempts - successes + 1
    lower_b, upper_b = stats.beta.interval(confidence, alpha, beta)
    return (lower_b, upper_b)


def plot_abresist_frac_error(
    df,
    ex,
    year,
    verbose=True,
    sims=100,
    figname="doodoo",
    savefig=False,
    smooth=5,
    col="Drug Class",
    value="phenotype",
):

    sns.set_theme(font="Arial", style="white", font_scale=0.7)

    frac_df, pre, post = measure_obs_distance(df, ex, year, col, verbose)
    dist = post - pre

    print(frac_df)

    # histogram of fraction of strains w/ RGI hits for drug class
    plt.figure(figsize=(14, 8))
    chart = sns.barplot(
        data=frac_df, x="year", y="frac", color="salmon", saturation=0.5
    )
    chart.bar_label(chart.containers[0])

    plt.axvline(pd.Index(frac_df["year"]).get_loc(year, method="nearest"))
    plt.ylabel("Fraction of bugs with phenotype")
    plt.xlabel("Year")
    plt.xticks(rotation=45)

    plt.show()

    d = {"Year": [], "num_pos": [], "all": [], "Anthropogenicity": []}
    for years in range(frac_df["year"].min(), frac_df["year"].max()):
        upb = smooth + years
        downb = years - smooth
        g = frac_df.loc[(frac_df["year"] <= upb) & (downb <= frac_df["year"])]
        if years not in frac_df["year"].values:
            continue
        d["Year"].append(years)
        if years >= year:
            d["Anthropogenicity"].append("Post-Human")
        else:
            d["Anthropogenicity"].append("Pre-Human")
        d["num_pos"].append(g["num_pos"].sum())
        d["all"].append(g["all"].sum())
    xdf = pd.DataFrame(data=d)
    print(xdf)
    xdf["frac"] = xdf["num_pos"] / xdf["all"]
    xdf[["lower-error", "upper-error"]] = xdf.apply(
        calculate_frequency_bounds, axis=1, result_type="expand"
    )
    if ~xdf['Anthropogenicity'].isin(['Pre-Human']).any():
        return 'nan'

    # make subplots for the figure
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    ax1.set(xlabel="Year Cultured")

    lp = sns.lineplot(
        data=xdf,
        x="Year",
        y="frac",
        markers=True,
        hue="Anthropogenicity",
        palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[1]],
        ax=ax1,
    )
    lp.set(ylim=(0, 1))

    # Get the current legend
    legend = ax1.legend()

    # Change the legend title
    legend.set_title("")

    # Change legend labels
    new_labels = ["Pre-clinical", "Post-clinical"]
    for t, l in zip(
        legend.texts[0:], new_labels
    ):  
        t.set_text(l)

    dp = sns.scatterplot(
        data=xdf,
        x="Year",
        y="frac",
        hue="Anthropogenicity",
        palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[1]],
        legend=False,
        ax=ax1,
        edgecolor="none",
    )
    plt.fill_between(
        xdf.loc[xdf["Year"] < year]["Year"],
        xdf.loc[xdf["Year"] < year]["lower-error"],
        xdf.loc[xdf["Year"] < year]["upper-error"],
        alpha=0.3,
        edgecolor=sns.color_palette("deep")[0],
        facecolor=sns.color_palette("deep")[0],
    )
    plt.fill_between(
        xdf.loc[xdf["Year"] >= year]["Year"],
        xdf.loc[xdf["Year"] >= year]["lower-error"],
        xdf.loc[xdf["Year"] >= year]["upper-error"],
        alpha=0.3,
        edgecolor=sns.color_palette("deep")[1],
        facecolor=sns.color_palette("deep")[1],
    )
    sns.despine()
    plt.axvline(year, color="red")
    ax1.set(ylabel="Fraction of isolates with {}".format(value))
    plt.axvline(year, color="red")  # , label='{} introduced'.format(year))
    trans = ax1.get_xaxis_transform()
    plt.text(
        year - 4,
        0.3,
        "Introduced clinically in {}".format(year),
        rotation=90,
        transform=trans,
        fontsize="small",
    )
    preab_df = xdf.loc[xdf["Anthropogenicity"] == "Pre-Human"]
    lp.hlines(
        y=preab_df["frac"].mean(),
        xmin=preab_df["Year"].min(),
        xmax=preab_df["Year"].max(),
        color=sns.color_palette("deep")[0],
    )

    postab_df = xdf.loc[xdf["Anthropogenicity"] == "Post-Human"]
    lp.hlines(
        y=postab_df["frac"].mean(),
        xmin=postab_df["Year"].min(),
        xmax=postab_df["Year"].max(),
        color=sns.color_palette("deep")[1],
    )
    if savefig:
        plt.gcf().set_size_inches(5.0, 3)
        plt.savefig(
            "{}-fractionofresist.svg".format(figname),
            bbox_inches="tight",
            dpi=300,
        )

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.set(ylabel="Count of isolates analyzed", xlabel="Year Cultured")
    # create another copy of the dataframe
    drop_dups = df.drop_duplicates(subset=["Accession_Number"]).copy()
    drop_dups["Anthropogenicity"] = "Post-Human"

    drop_dups.loc[drop_dups["Year_Cultured"] < year, ["Anthropogenicity"]] = "Pre-Human"
    try:
        histogram = sns.histplot(
            data=drop_dups,
            x="Year_Cultured",
            hue="Anthropogenicity",
            palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[1]],
            ax=ax2,
            cumulative=True,
            fill=False,
            element="step",
        )
    except np.core._exceptions._ArrayMemoryError:
        histogram = sns.histplot(
            data=drop_dups,
            x="Year_Cultured",
            hue="Anthropogenicity",
            palette=[sns.color_palette("deep")[0], sns.color_palette("deep")[1]],
            ax=ax2,
            cumulative=True,
            fill=False,
            element="step",
            bins='sturges'
        )
    sns.despine()
    if savefig:
        plt.gcf().set_size_inches(5.0, 3)
        plt.savefig(
            "{}-histogram.svg".format(figname),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()

    if sims == 0:
        return

    preab_mean = frac_df.loc[frac_df["Anthropogenicity"] == "Pre-Human"]["frac"].mean()
    postab_mean = frac_df.loc[frac_df["Anthropogenicity"] == "Post-Human"][
        "frac"
    ].mean()

    nulldist = shuffleyears_structured(df, ex, year, col, verbose, simulations=sims)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    try:
        sns.histplot(
            nulldist, stat="density", color=sns.color_palette("deep")[0], linewidth=0
        )
    except np.core._exceptions._ArrayMemoryError:
        sns.histplot(
            nulldist, stat="density", color=sns.color_palette("deep")[0], 
            bins=200,
            linewidth=0
        )
    xmin, xmax = plt.xlim()
    mu_struct, std_struct = norm.fit(nulldist)
    x = np.linspace(xmin, xmax, sims)
    y_pdf = norm.pdf(x, mu_struct, std_struct)
    plt.plot(x, y_pdf, "k", linewidth=2)
    plt.axvline(dist, color="red")
    trans = ax.get_xaxis_transform()
    # x = 10
    plt.text(
        dist * 0.85,
        0.5,
        "Observed $\Delta$ = {:.3f}".format(dist),
        rotation=90,
        transform=trans,
        fontsize="x-small",
    )
    plt.title("Structured Shuffling")
    plt.xlabel("Null $\Delta$ distribution in {} simulations".format(sims))
    sns.despine()
    pval_struct = norm.cdf(dist, mu_struct, std_struct)
    if pval_struct > 0.5:
        pval_struct = 1 - pval_struct
    ax = plt.gca()
    ax.text(0.95, 0.95, "p-value = {:.3f}".format(pval_struct), transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))

    if savefig:
        plt.gcf().set_size_inches(2, 3)
        plt.savefig(
            "{}-structuredshuffling.svg".format(figname), bbox_inches="tight", dpi=300
        )
    plt.show()
    return pval_struct

    nulldist = shuffleyears_unstructured(df, ex, year, col, verbose, simulations=sims)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.histplot(
        nulldist, stat="density", color=sns.color_palette("deep")[0], edgecolor="none"
    )
    xmin, xmax = plt.xlim()
    mu_unstruct, std_unstruct = norm.fit(nulldist)
    x = np.linspace(xmin, xmax, sims)
    y_pdf = norm.pdf(x, mu_unstruct, std_unstruct)
    plt.plot(x, y_pdf, "k", linewidth=2)
    plt.axvline(dist, color="red")
    trans = ax.get_xaxis_transform()
    plt.text(
        dist * 0.85,
        0.5,
        "Observed $\Delta$ = {:.3f}".format(dist),
        rotation=90,
        transform=trans,
        fontsize="x-small",
    )
    plt.title("Unstructured Shuffling")
    plt.xlabel("Null $\Delta$ distribution in {} simulations".format(sims))
    sns.despine()
    pval = norm.cdf(dist, mu_struct, std_struct)
    if pval > 0.5:
        pval = 1 - pval
    ax = plt.gca()
    ax.text(0.95, 0.95, "p-value = {:.3f}".format(pval), transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
    if savefig:
        plt.gcf().set_size_inches(2, 3)
        plt.savefig(
            "{}-unstructuredshuffling.svg".format(figname), bbox_inches="tight", dpi=300
        )
    plt.show()
    pval = norm.cdf(dist, mu_unstruct, std_unstruct)
    if pval > 0.5:
        pval = 1 - pval
    print("P-value (unstructured) = {:.4f}".format(pval))

drugyear_usage_amrfinder = {
    "FOSFOMYCIN": 1971,
    "BETA-LACTAM": 1943,
    "PHENICOL/QUINOLONE": 1949,
    "TETRACYCLINE": 1948,
    "CEPHALOSPORIN": 1964,
    "COLISTIN": 1959,
    "SULFONAMIDE": 1936,
    "QUINOLONE": 1962,
    "STREPTOMYCIN": 1948, # when first clinical trial ended
    "CHLORAMPHENICOL": 1949,
    "KANAMYCIN": 1957, # when first described in lit
    "MACROLIDE": 1952,
    "FLUOROQUINOLONE": 1962,
    "TRIMETHOPRIM": 1962,
    # "RIFAMYCIN": 1963,
    # "AMINOGLYCOSIDE": 1946,
    # "COLISTIN": 1959,
    # "LINCOSAMIDE": 1963,
    # "GLYCOPEPTIDE": 1958,
    # "FUSIDIC ACID": 1962,
}

drugyear_usage_rgi = {
    "fluoroquinolone antibiotic": 1962,
    "penam": 1943,
    "cephalosporin": 1964,
    "tetracycline antibiotic": 1948,
    "phenicol antibiotic": 1949,
    "macrolide antibiotic": 1952,
    "rifamycin antibiotic": 1963,
    "aminoglycoside antibiotic": 1946,
    # "peptide antibiotic": 1941,  # not sure about this one
    # "glycylcycline": 1948,  # using year tetracyclines were introduced clinically
    # "triclosan": 1968,  # using wiki page
    # "cephamycin": 1964,  # using cephalosporin year
    "carbapenem": 1985,
    # "aminocoumarin antibiotic": 1965,  # best guess from wiki article
    # "penem": 1985,  # using year carbapenems were introduced
    "monobactam": 1986,
    # "disinfecting agents and intercalating dyes": 1930,  # no clue
    # "acridine dye": 1970,  # no clue?
    # "diaminopyrimidine antibiotic": 1962,
    # "elfamycin antibiotic": 1978,  # no clue
    # "fosfomycin": 1971,
    # "nucleoside antibiotic": 2014,  # no clue, but looks newish
    "lincosamide antibiotic": 1963,
    # "nitroimidazole antibiotic": 1960,
    # "Null": 1920,
    # "benzalkonium chloride": 1950,  # no clude
    # "rhodamine": 1950,  # no clude
    "sulfonamide antibiotic": 1936,
    "nitrofuran antibiotic": 1953,
    # "streptogramin antibiotic": 1965,
    # "oxazolidinone antibiotic": 2000,
    "glycopeptide antibiotic": 1958,
    # "fusidic acid": 1962,
    # "pleuromutilin antibiotic": 2007,
    # "bicyclomycin": 1972,  # from wiki
    # "antibacterial free fatty acids": 2000,  # noclude
    # "para-aminosalicylic acid": 1943,
    # "isoniazid": 1952,
    # "polyamine antibiotic": 2005,  # no idea
}

eskape_pathogens = df.loc[
    df["Strain"].str.contains(
        "Enterococcus faecium|Salmonella enterica|Klebsiella pneumoniae|Acinetobacter baumannii|Pseudomonas aeruginosa|Enterobacter"
    )
].copy()
eskape_pathogens["CleanSpecies"] = eskape_pathogens["Strain"]
eskape_pathogens.loc[
    eskape_pathogens["Strain"].str.contains("Enterococcous faecium"), ["CleanSpecies"]
] = "Enterococcus faecium"
eskape_pathogens.loc[
    eskape_pathogens["Strain"].str.contains("Salmonella enterica"), ["CleanSpecies"]
] = "Salmonella enterica"
eskape_pathogens.loc[
    eskape_pathogens["Strain"].str.contains("Klebsiella pneumoniae"), ["CleanSpecies"]
] = "Klebsiella pneumoniae"
eskape_pathogens.loc[
    eskape_pathogens["Strain"].str.contains("Acinetobacter baumannii"), ["CleanSpecies"]
] = "Acinetobacter baumannii"
eskape_pathogens.loc[
    eskape_pathogens["Strain"].str.contains("Pseudomonas aeruginosa"), ["CleanSpecies"]
] = "Pseudomonas aeruginosa"
eskape_pathogens.loc[
    eskape_pathogens["Strain"].str.contains("Enterobacter"), ["CleanSpecies"]
] = "Enterobacter sp."
eskape_pathogens.drop_duplicates("Accession_Number").sort_values("Year_Cultured")
family_name = eskape_pathogens["CleanSpecies"].str.split(" ", expand=True)[0].str[0]
species_name = eskape_pathogens["CleanSpecies"].str.split(" ", expand=True)[1]
eskape_pathogens["Species"] = family_name + ". " + species_name
eskape_pathogens.loc[
    eskape_pathogens["Species"].str.match("E. sp."), ["Species"]
] = "Enterobacter sp."

amrfinder_df = pd.read_csv(
    "../../data/5_amrfinderplus/merged_amrfinderout",
    sep="\t",
    names=[
        "Accession_Number",
        "Protein identifier",
        "Contig id",
        "Start",
        "Stop",
        "Strand",
        "Gene symbol",
        "Sequence name",
        "Scope",
        "Element type",
        "Element subtype",
        "Class",
        "Subclass",
        "Method",
        "Target length",
        "Reference sequence length",
        "Coverage of reference sequence",
        "% Identity to reference sequence",
        "Alignment length",
        "Accession of closest sequence",
        "Name of closest sequence",
        "HMM id",
        "HMM description",
    ],
)
amrfinder_df = pd.merge(amrfinder_df, df.drop_duplicates("Accession_Number")[["Accession_Number", "Species", "Year_Cultured"]],
                     on='Accession_Number', how='left')
# amrfinder_df['Species'].value_counts()
ecoli_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Escherichia coli')]
# pyogenes_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('pyogenes')]
staph_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Staphylococcus aureus')]
kleb_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Klebsiella pneumoniae')]
pseud_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Pseudomonas aeruginosa')]
salm_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Salmonella enterica')]
acineto_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Acinetobacter baumannii')]
enterobacter_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Enterobacter')]
faecium_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('faecium')]
streppneumo_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('Streptococcus pneumoniae')]
eskape_amrfinderdf = amrfinder_df.loc[amrfinder_df['Species'].str.contains('faecium|Enterobacter|Acinetobacter baumannii|Klebsiella pneumoniae|Salmonella enterica')]

sims = 10000
for d in drugyear_usage_amrfinder:
    drug_name = d.strip().replace('/', '-')
    print(drug_name)
    class_val=False
    subclass_val = False
    if amrfinder_df['Class'].isin([d]).any():
        class_val = True
    if amrfinder_df['Subclass'].isin([d]).any():
        subclass_val = True
    with open('./figs/fig4/fig4c/amrfinder/all/pvals-all', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = amrfinder_df,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/all/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = amrfinder_df,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/all/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("All bugs completed")
    
    with open('./figs/fig4/fig4c/amrfinder/acinetobacter/pvals-acineto', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = acineto_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/acinetobacter/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = acineto_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/acinetobacter/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("Acinetobacter complete")
    
    
    with open('./figs/fig4/fig4c/amrfinder/ecoli/pvals-ecoli', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = ecoli_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/ecoli/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = ecoli_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/ecoli/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("Ecoli complete")
        
    with open('./figs/fig4/fig4c/amrfinder/enterobacter/pvals-enterobacter', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = enterobacter_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/enterobacter/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = enterobacter_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/enterobacter/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("Enterobacter complete")
    
    with open('./figs/fig4/fig4c/amrfinder/enterococcusfaecium/pvals-faecium', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = faecium_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/enterococcusfaecium/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = faecium_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/enterococcusfaecium/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("Faecium complete")
        
    with open('./figs/fig4/fig4c/amrfinder/eskape/pvals-eskape', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = eskape_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/eskape/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = eskape_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/eskape/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("ESKAPE complete")
    
            
    with open('./figs/fig4/fig4c/amrfinder/kleb/pvals-kleb', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = kleb_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/kleb/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = kleb_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/kleb/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("Kleb complete")
    
    
                
    with open('./figs/fig4/fig4c/amrfinder/pseud/pvals-pseud', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = pseud_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/pseud/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = pseud_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/pseud/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("Pseud complete")
    
                
    with open('./figs/fig4/fig4c/amrfinder/salm/pvals-salm', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = salm_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/salm/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = salm_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/salm/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("salm complete")
    
                
    with open('./figs/fig4/fig4c/amrfinder/staph/pvals-staph', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = staph_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/staph/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = staph_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/staph/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("staph complete")
    
                    
    with open('./figs/fig4/fig4c/amrfinder/streppneumo/pvals-streppneumo', 'a') as f:
        if class_val:
            pval = plot_abresist_frac_error(
                df = streppneumo_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Class',
                figname=f"./figs/fig4/fig4c/amrfinder/streppneumo/{drug_name}"
            )
        else:
            pval = plot_abresist_frac_error(
                df = streppneumo_amrfinderdf,
                ex=d,
                year=drugyear_usage_amrfinder[d],
                sims=sims,
                verbose=False,
                value="≥ 1 resistance-associated element",
                smooth=5,
                savefig=True,
                col='Subclass',
                figname=f"./figs/fig4/fig4c/amrfinder/streppneumo/{drug_name}"
            )
        f.write(f"{drug_name}\t{drugyear_usage_amrfinder[d]}\t{pval}\n")
    print("streppneumo complete")
# with open('./figs/fig4/fig4c


# with open('pvals-all-all', 'w') as f:
#     for d in drugyear_usage_rgi:
#         stripped=d.strip()
#         pval = plot_abresist_frac_error(
#             df=df,
#             ex=d,
#             year=drugyear_usage[d],
#             sims=5000,
#             verbose=False,
#             col="Drug Class",
#             value="≥ 1 resistance-associated element",
#             savefig=True,
#             smooth=5,
#             figname=f"./figs/fig4/fig4c/all-all/{stripped}",
#         )
        # f.write(f"{d}\t{drugyear_usage[d]}\t{pval}")

# with open('pvals-eskape-all', 'w') as f:
#     for d in drugyear_usage_rgi:
#         stripped=d.strip()
#         pval = plot_abresist_frac_error(
#             df=eskape_pathogens,
#             ex=d,
#             year=drugyear_usage[d],
#             sims=5000,
#             verbose=False,
#             col="Drug Class",
#             value="≥ 1 resistance-associated element",
#             savefig=True,
#             smooth=5,
#             figname=f"./figs/fig4/fig4c/eskape-all/{stripped}",
#         )
#         f.write(f"{d}\t{drugyear_usage[d]}\t{pval}")

