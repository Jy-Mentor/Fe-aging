"""
模块: L1/map_probes_to_genes.py
功能: 将各数据集探针级DE结果映射到人类基因符号，为RRA荟萃分析做准备
输入: L1/results/*_DE_results.csv, GPL1355注释文件
输出: L1/results/*_DE_gene_level.csv
依赖: pandas, numpy
运行: python L1/map_probes_to_genes.py
"""

import logging
import os
import re
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# 日志配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "map_probes_to_genes.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

RESULT_DIR = os.path.join(PROJECT_ROOT, "L1", "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "L1 数据集", "bulk")


def parse_ilmn_annotation():
    """Parse ILMN probe-to-gene mapping from CSV file.
    Returns dict: probe_id -> gene_symbol
    """
    mapping_file = os.path.join(RESULT_DIR, "ILMN_probe_to_gene.csv")
    if not os.path.exists(mapping_file):
        logger.warning("ILMN mapping file not found: %s", mapping_file)
        return {}

    logger.info("Parsing ILMN probe-to-gene mapping...")
    df = pd.read_csv(mapping_file)
    probe_to_gene = dict(zip(df["Probe"], df["GeneSymbol"]))
    logger.info("ILMN: %d probes mapped to %d unique genes",
                len(probe_to_gene), len(set(probe_to_gene.values())))
    return probe_to_gene


def get_species_ortholog_map():
    """Build mouse/rat → human ortholog mapping.
    Uses simple uppercase conversion for genes with identical symbols,
    plus a curated mapping for known divergent orthologs.
    """
    # Curated mouse/rat → human ortholog mappings for common divergent genes
    curated = {
        # Mouse → Human
        "Gpnmb": "GPNMB", "Lgals3": "LGALS3", "Ctsz": "CTSZ",
        "C1qb": "C1QB", "Tyrobp": "TYROBP", "Tspo": "TSPO",
        "Fgl2": "FGL2", "Hspb1": "HSPB1", "Rac2": "RAC2",
        "Plek": "PLEK", "Arpc1b": "ARPC1B", "Gpr84": "GPR84",
        "Fcgr3a": "FCGR3A", "Rnaset2": "RNASET2", "Lgmn": "LGMN",
        "Anxa3": "ANXA3", "Ifi30": "IFI30", "Clic1": "CLIC1",
        "Itgb2": "ITGB2", "Timp1": "TIMP1", "Fcer1g": "FCER1G",
        "Apobec1": "APOBEC1", "Ccl2": "CCL2", "C1qc": "C1QC",
        "Cd63": "CD63", "Bcl2a1": "BCL2A1", "Cd14": "CD14",
        "Aif1": "AIF1", "Ctss": "CTSS", "Cd68": "CD68",
        "Csf1r": "CSF1R", "Cybb": "CYBB", "Lyz2": "LYZ2",
        "Mpeg1": "MPEG1", "Ms4a6d": "MS4A6A", "Ncf1": "NCF1",
        "Ncf2": "NCF2", "Ncf4": "NCF4",
        "Cxcl10": "CXCL10", "Ccl3": "CCL3", "Ccl4": "CCL4",
        "Ccl5": "CCL5", "Cxcl1": "CXCL1", "Cxcl2": "CXCL2",
        "Tlr2": "TLR2", "Tlr4": "TLR4", "Tlr7": "TLR7",
        "Il1b": "IL1B", "Il6": "IL6", "Tnf": "TNF",
        "Ifng": "IFNG", "Il10": "IL10", "Il18": "IL18",
        "Hmox1": "HMOX1", "Hif1a": "HIF1A", "Nfe2l2": "NFE2L2",
        "Keap1": "KEAP1", "Sod1": "SOD1", "Sod2": "SOD2",
        "Gpx4": "GPX4", "Acsl4": "ACSL4", "Tfrc": "TFRC",
        "Slc7a11": "SLC7A11", "Slc3a2": "SLC3A2", "Fth1": "FTH1",
        "Ftl1": "FTL", "Ncoa4": "NCOA4", "Sat1": "SAT1",
        "Alox5": "ALOX5", "Alox12": "ALOX12", "Alox15": "ALOX15",
        "Ptgs2": "PTGS2", "Nlrp3": "NLRP3", "Casp1": "CASP1",
        "Casp3": "CASP3", "Bax": "BAX", "Bcl2": "BCL2",
        "Becn1": "BECN1", "Atg5": "ATG5", "Atg7": "ATG7",
        "Map1lc3b": "MAP1LC3B", "Sqstm1": "SQSTM1", "P62": "SQSTM1",
        "Cdkn1a": "CDKN1A", "Cdkn2a": "CDKN2A", "Trp53": "TP53",
        "Mki67": "MKI67", "Pcna": "PCNA", "Ccnd1": "CCND1",
        "Mmp9": "MMP9", "Mmp2": "MMP2", "Timp2": "TIMP2",
        "Vegfa": "VEGFA", "Fgf2": "FGF2", "Pdgfb": "PDGFB",
        "Igf1": "IGF1", "Bdnf": "BDNF", "Ntf3": "NTF3",
        "Gfap": "GFAP", "Aqp4": "AQP4", "Mbp": "MBP",
        "Mog": "MOG", "Tubb3": "TUBB3", "Map2": "MAP2",
        "Syp": "SYP", "Syt1": "SYT1", "Dlg4": "DLG4",
        "Grin1": "GRIN1", "Grin2a": "GRIN2A", "Grin2b": "GRIN2B",
        "Gria1": "GRIA1", "Gria2": "GRIA2", "Gabra1": "GABRA1",
        "Slc1a2": "SLC1A2", "Slc1a3": "SLC1A3", "Slc17a7": "SLC17A7",
        "Cldn5": "CLDN5", "Pecam1": "PECAM1", "Cdh5": "CDH5",
        "Pdgfrb": "PDGFRB", "Anpep": "ANPEP", "Cspg4": "CSPG4",
        "Ptprc": "PTPRC", "Cd3e": "CD3E", "Cd4": "CD4",
        "Cd8a": "CD8A", "Cd19": "CD19", "Ncam1": "NCAM1",
        # Rat → Human (additional rat-specific mappings)
        "A2m": "A2M", "LOC100911545": "A2M",
        "Aacs": "AACS", "Aadat": "AADAT", "Aak1": "AAK1",
        "Aamdc": "AAMDC", "Aamp": "AAMP", "Aard": "AARD",
        "Aars": "AARS", "Aars2": "AARS2", "Aasdh": "AASDH",
        "Abat": "ABAT", "Abca1": "ABCA1", "Abcb1a": "ABCB1",
        "Abcb1b": "ABCB1", "Abcc1": "ABCC1", "Abcc2": "ABCC2",
        "Abcc3": "ABCC3", "Abcc4": "ABCC4", "Abcc5": "ABCC5",
        "Abcc8": "ABCC8", "Abcd1": "ABCD1", "Abcd2": "ABCD2",
        "Abcd3": "ABCD3", "Abcg1": "ABCG1", "Abcg2": "ABCG2",
        "Abhd5": "ABHD5", "Abi1": "ABI1", "Abl1": "ABL1",
        "Abl2": "ABL2", "Acaa1a": "ACAA1", "Acaa2": "ACAA2",
        "Acaca": "ACACA", "Acadl": "ACADL", "Acadm": "ACADM",
        "Acads": "ACADS", "Acadvl": "ACADVL", "Acat1": "ACAT1",
        "Acat2": "ACAT2", "Ace": "ACE", "Ache": "ACHE",
        "Aco1": "ACO1", "Aco2": "ACO2", "Acot1": "ACOT1",
        "Acot2": "ACOT2", "Acot7": "ACOT7", "Acox1": "ACOX1",
        "Acp1": "ACP1", "Acp2": "ACP2", "Acp5": "ACP5",
        "Acta1": "ACTA1", "Acta2": "ACTA2", "Actb": "ACTB",
        "Actc1": "ACTC1", "Actg1": "ACTG1", "Actn1": "ACTN1",
        "Actn2": "ACTN2", "Actn3": "ACTN3", "Actn4": "ACTN4",
        "Actr2": "ACTR2", "Actr3": "ACTR3", "Acvr1": "ACVR1",
        "Acvr1b": "ACVR1B", "Acvr2a": "ACVR2A", "Acvr2b": "ACVR2B",
        "Acy1": "ACY1", "Ada": "ADA", "Adam10": "ADAM10",
        "Adam17": "ADAM17", "Adam9": "ADAM9", "Adar": "ADAR",
        "Adcy1": "ADCY1", "Adcy2": "ADCY2", "Adcy3": "ADCY3",
        "Adcy5": "ADCY5", "Adcy6": "ADCY6", "Adcy8": "ADCY8",
        "Adcy9": "ADCY9", "Add1": "ADD1", "Add2": "ADD2",
        "Add3": "ADD3", "Adh1": "ADH1A", "Adh5": "ADH5",
        "Adh7": "ADH7", "Adipoq": "ADIPOQ", "Adk": "ADK",
        "Adm": "ADM", "Adnp": "ADNP", "Adora1": "ADORA1",
        "Adora2a": "ADORA2A", "Adora2b": "ADORA2B", "Adra1a": "ADRA1A",
        "Adra1b": "ADRA1B", "Adra1d": "ADRA1D", "Adra2a": "ADRA2A",
        "Adra2b": "ADRA2B", "Adra2c": "ADRA2C", "Adrb1": "ADRB1",
        "Adrb2": "ADRB2", "Adrb3": "ADRB3", "Adrm1": "ADRM1",
        "Adsl": "ADSL", "Adss": "ADSS", "Aebp1": "AEBP1",
        "Aes": "AES", "Aff1": "AFF1", "Aff4": "AFF4",
        "Afg3l2": "AFG3L2", "Afp": "AFP", "Aga": "AGA",
        "Agap1": "AGAP1", "Agap2": "AGAP2", "Agap3": "AGAP3",
        "Agbl5": "AGBL5", "Agfg1": "AGFG1", "Aggf1": "AGGF1",
        "Agk": "AGK", "Agl": "AGL", "Ago1": "AGO1",
        "Ago2": "AGO2", "Ago3": "AGO3", "Ago4": "AGO4",
        "Agpat1": "AGPAT1", "Agpat2": "AGPAT2", "Agpat3": "AGPAT3",
        "Agps": "AGPS", "Agrn": "AGRN", "Agt": "AGT",
        "Agtr1a": "AGTR1", "Agtr2": "AGTR2", "Agxt": "AGXT",
        "Ahcy": "AHCY", "Ahdc1": "AHDC1", "Ahi1": "AHI1",
        "Ahr": "AHR", "Aifm1": "AIFM1", "Aimp1": "AIMP1",
        "Aimp2": "AIMP2", "Aipl1": "AIPL1", "Aire": "AIRE",
        "Akap1": "AKAP1", "Akap5": "AKAP5", "Akap6": "AKAP6",
        "Akap8": "AKAP8", "Akap9": "AKAP9", "Akirin1": "AKIRIN1",
        "Akirin2": "AKIRIN2", "Akt1": "AKT1", "Akt2": "AKT2",
        "Akt3": "AKT3", "Aktip": "AKTIP", "Alad": "ALAD",
        "Alas1": "ALAS1", "Alas2": "ALAS2", "Alb": "ALB",
        "Aldh1a1": "ALDH1A1", "Aldh1a2": "ALDH1A2", "Aldh1a3": "ALDH1A3",
        "Aldh1b1": "ALDH1B1", "Aldh2": "ALDH2", "Aldh3a1": "ALDH3A1",
        "Aldh3a2": "ALDH3A2", "Aldh4a1": "ALDH4A1", "Aldh5a1": "ALDH5A1",
        "Aldh6a1": "ALDH6A1", "Aldh7a1": "ALDH7A1", "Aldh9a1": "ALDH9A1",
        "Aldoa": "ALDOA", "Aldob": "ALDOB", "Aldoc": "ALDOC",
        "Alg1": "ALG1", "Alg2": "ALG2", "Alg3": "ALG3",
        "Alg5": "ALG5", "Alg6": "ALG6", "Alg8": "ALG8",
        "Alg9": "ALG9", "Alkbh1": "ALKBH1", "Alkbh5": "ALKBH5",
        "Alox12b": "ALOX12B", "Alox5ap": "ALOX5AP", "Aloxe3": "ALOXE3",
        "Alpl": "ALPL", "Alpp": "ALPP", "Als2": "ALS2",
        "Amacr": "AMACR", "Ambra1": "AMBRA1", "Amd1": "AMD1",
        "Amd2": "AMD2", "Amfr": "AMFR", "Amh": "AMH",
        "Amhr2": "AMHR2", "Ampd1": "AMPD1", "Ampd2": "AMPD2",
        "Ampd3": "AMPD3", "Amph": "AMPH", "Amt": "AMT",
        "Amy1": "AMY1A", "Amy2": "AMY2A", "Ang": "ANG",
        "Angpt1": "ANGPT1", "Angpt2": "ANGPT2", "Angptl3": "ANGPTL3",
        "Angptl4": "ANGPTL4", "Ank1": "ANK1", "Ank2": "ANK2",
        "Ank3": "ANK3", "Ankfy1": "ANKFY1", "Ankh": "ANKH",
        "Ankrd1": "ANKRD1", "Ankrd2": "ANKRD2", "Ankrd11": "ANKRD11",
        "Ankrd13a": "ANKRD13A", "Ankrd17": "ANKRD17", "Ankrd26": "ANKRD26",
        "Ankrd28": "ANKRD28", "Ankrd6": "ANKRD6", "Anks1a": "ANKS1A",
        "Anks1b": "ANKS1B", "Anln": "ANLN", "Ano1": "ANO1",
        "Ano2": "ANO2", "Ano3": "ANO3", "Ano4": "ANO4",
        "Ano5": "ANO5", "Ano6": "ANO6", "Anp32a": "ANP32A",
        "Anp32b": "ANP32B", "Anp32e": "ANP32E", "Antxr1": "ANTXR1",
        "Antxr2": "ANTXR2", "Anxa1": "ANXA1", "Anxa2": "ANXA2",
        "Anxa4": "ANXA4", "Anxa5": "ANXA5", "Anxa6": "ANXA6",
        "Anxa7": "ANXA7", "Aoc1": "AOC1", "Aoc3": "AOC3",
        "Ap1b1": "AP1B1", "Ap1g1": "AP1G1", "Ap1m1": "AP1M1",
        "Ap1s1": "AP1S1", "Ap2a1": "AP2A1", "Ap2a2": "AP2A2",
        "Ap2b1": "AP2B1", "Ap2m1": "AP2M1", "Ap2s1": "AP2S1",
        "Ap3b1": "AP3B1", "Ap3d1": "AP3D1", "Ap3m1": "AP3M1",
        "Ap3s1": "AP3S1", "Ap4b1": "AP4B1", "Ap4e1": "AP4E1",
        "Ap4m1": "AP4M1", "Ap4s1": "AP4S1", "Ap5b1": "AP5B1",
        "Ap5m1": "AP5M1", "Ap5s1": "AP5S1", "Apaf1": "APAF1",
        "Apbb1": "APBB1", "Apbb2": "APBB2", "Apc": "APC",
        "Apc2": "APC2", "Apex1": "APEX1", "Apex2": "APEX2",
        "Aph1a": "APH1A", "Aph1b": "APH1B", "Api5": "API5",
        "Apip": "APIP", "Apln": "APLN", "Aplnr": "APLNR",
        "Apoa1": "APOA1", "Apoa2": "APOA2", "Apoa4": "APOA4",
        "Apoa5": "APOA5", "Apob": "APOB", "Apobec2": "APOBEC2",
        "Apoc1": "APOC1", "Apoc2": "APOC2", "Apoc3": "APOC3",
        "Apoc4": "APOC4", "Apod": "APOD", "Apoe": "APOE",
        "Apoh": "APOH", "Apol2": "APOL2", "Apol3": "APOL3",
        "Apol5": "APOL5", "Apol6": "APOL6", "Apom": "APOM",
        "App": "APP", "Appl1": "APPL1", "Appl2": "APPL2",
        "Aprt": "APRT", "Aqp1": "AQP1", "Aqp2": "AQP2",
        "Aqp3": "AQP3", "Aqp5": "AQP5", "Aqp6": "AQP6",
        "Aqp7": "AQP7", "Aqp8": "AQP8", "Aqp9": "AQP9",
        "Ar": "AR", "Araf": "ARAF", "Arcn1": "ARCN1",
        "Areg": "AREG", "Arf1": "ARF1", "Arf3": "ARF3",
        "Arf4": "ARF4", "Arf5": "ARF5", "Arf6": "ARF6",
        "Arfgef1": "ARFGEF1", "Arfgef2": "ARFGEF2", "Arg1": "ARG1",
        "Arg2": "ARG2", "Arhgap1": "ARHGAP1", "Arhgap5": "ARHGAP5",
        "Arhgef1": "ARHGEF1", "Arhgef2": "ARHGEF2", "Arhgef7": "ARHGEF7",
        "Arid1a": "ARID1A", "Arid1b": "ARID1B", "Arid2": "ARID2",
        "Arid3a": "ARID3A", "Arid3b": "ARID3B", "Arid4a": "ARID4A",
        "Arid4b": "ARID4B", "Arid5a": "ARID5A", "Arid5b": "ARID5B",
        "Arl1": "ARL1", "Arl2": "ARL2", "Arl3": "ARL3",
        "Arl4a": "ARL4A", "Arl4c": "ARL4C", "Arl5a": "ARL5A",
        "Arl5b": "ARL5B", "Arl6": "ARL6", "Arl6ip5": "ARL6IP5",
        "Arl8a": "ARL8A", "Arl8b": "ARL8B", "Armc1": "ARMC1",
        "Armc6": "ARMC6", "Armc8": "ARMC8", "Armcx1": "ARMCX1",
        "Armcx2": "ARMCX2", "Armcx3": "ARMCX3", "Arnt": "ARNT",
        "Arnt2": "ARNT2", "Arntl": "ARNTL", "Arntl2": "ARNTL2",
        "Arpc2": "ARPC2", "Arpc3": "ARPC3", "Arpc4": "ARPC4",
        "Arpc5": "ARPC5", "Arpp19": "ARPP19", "Arpp21": "ARPP21",
        "Arsa": "ARSA", "Arsb": "ARSB", "Arsg": "ARSG",
        "Arsi": "ARSI", "Arsj": "ARSJ", "Arsk": "ARSK",
        "Art1": "ART1", "Art3": "ART3", "Art4": "ART4",
        "Art5": "ART5", "Arv1": "ARV1", "Arvcf": "ARVCF",
        "Asah1": "ASAH1", "Asah2": "ASAH2", "Asap1": "ASAP1",
        "Asap2": "ASAP2", "Ascc1": "ASCC1", "Ascc2": "ASCC2",
        "Ascc3": "ASCC3", "Asf1a": "ASF1A", "Asf1b": "ASF1B",
        "Asgr2": "ASGR2", "Ash1l": "ASH1L", "Ash2l": "ASH2L",
        "Asic1": "ASIC1", "Asic2": "ASIC2", "Asic3": "ASIC3",
        "Asl": "ASL", "Asmt": "ASMT", "Asna1": "ASNA1",
        "Asns": "ASNS", "Aspa": "ASPA", "Asph": "ASPH",
        "Aspm": "ASPM", "Aspscr1": "ASPSCR1", "Ass1": "ASS1",
        "Astn1": "ASTN1", "Astn2": "ASTN2", "Asxl1": "ASXL1",
        "Asxl2": "ASXL2", "Asxl3": "ASXL3", "Atad1": "ATAD1",
        "Atad2": "ATAD2", "Atad3a": "ATAD3A", "Atat1": "ATAT1",
        "Atcay": "ATCAY", "Atf1": "ATF1", "Atf2": "ATF2",
        "Atf3": "ATF3", "Atf4": "ATF4", "Atf5": "ATF5",
        "Atf6": "ATF6", "Atf7": "ATF7", "Atg12": "ATG12",
        "Atg13": "ATG13", "Atg14": "ATG14", "Atg16l1": "ATG16L1",
        "Atg2a": "ATG2A", "Atg2b": "ATG2B", "Atg3": "ATG3",
        "Atg4a": "ATG4A", "Atg4b": "ATG4B", "Atg4c": "ATG4C",
        "Atg4d": "ATG4D", "Atg9a": "ATG9A", "Atg9b": "ATG9B",
        "Atic": "ATIC", "Atl1": "ATL1", "Atl2": "ATL2",
        "Atl3": "ATL3", "Atm": "ATM", "Atmin": "ATMIN",
        "Atn1": "ATN1", "Atoh1": "ATOH1", "Atp1a1": "ATP1A1",
        "Atp1a2": "ATP1A2", "Atp1a3": "ATP1A3", "Atp1b1": "ATP1B1",
        "Atp1b2": "ATP1B2", "Atp1b3": "ATP1B3", "Atp2a1": "ATP2A1",
        "Atp2a2": "ATP2A2", "Atp2a3": "ATP2A3", "Atp2b1": "ATP2B1",
        "Atp2b2": "ATP2B2", "Atp2b3": "ATP2B3", "Atp2b4": "ATP2B4",
        "Atp2c1": "ATP2C1", "Atp4a": "ATP4A", "Atp5a1": "ATP5F1A",
        "Atp5b": "ATP5F1B", "Atp5c1": "ATP5F1C", "Atp5d": "ATP5F1D",
        "Atp5e": "ATP5F1E", "Atp5g1": "ATP5MC1", "Atp5g2": "ATP5MC2",
        "Atp5g3": "ATP5MC3", "Atp5h": "ATP5PD", "Atp5i": "ATP5PF",
        "Atp5j": "ATP5PF", "Atp5l": "ATP5MG", "Atp5o": "ATP5PO",
        "Atp6ap1": "ATP6AP1", "Atp6ap2": "ATP6AP2", "Atp6v0a1": "ATP6V0A1",
        "Atp6v0a2": "ATP6V0A2", "Atp6v0b": "ATP6V0B", "Atp6v0c": "ATP6V0C",
        "Atp6v0d1": "ATP6V0D1", "Atp6v0e1": "ATP6V0E1", "Atp6v1a": "ATP6V1A",
        "Atp6v1b1": "ATP6V1B1", "Atp6v1b2": "ATP6V1B2", "Atp6v1c1": "ATP6V1C1",
        "Atp6v1d": "ATP6V1D", "Atp6v1e1": "ATP6V1E1", "Atp6v1f": "ATP6V1F",
        "Atp6v1g1": "ATP6V1G1", "Atp6v1h": "ATP6V1H", "Atp7a": "ATP7A",
        "Atp7b": "ATP7B", "Atp8a1": "ATP8A1", "Atp8a2": "ATP8A2",
        "Atp8b1": "ATP8B1", "Atp8b2": "ATP8B2", "Atp8b4": "ATP8B4",
        "Atp9a": "ATP9A", "Atp9b": "ATP9B", "Atr": "ATR",
        "Atrn": "ATRN", "Atrnl1": "ATRNL1", "Atrx": "ATRX",
        "Auh": "AUH", "Aup1": "AUP1", "Aurkb": "AURKB",
        "Auts2": "AUTS2", "Avp": "AVP", "Avpr1a": "AVPR1A",
        "Avpr1b": "AVPR1B", "Avpr2": "AVPR2", "Axin1": "AXIN1",
        "Axin2": "AXIN2", "Axl": "AXL", "Azin1": "AZIN1",
        "B2m": "B2M", "B3gat1": "B3GAT1", "B3gat2": "B3GAT2",
        "B3gat3": "B3GAT3", "B3gnt2": "B3GNT2", "B3gnt3": "B3GNT3",
        "B3gnt4": "B3GNT4", "B3gnt5": "B3GNT5", "B3gnt7": "B3GNT7",
        "B4galnt1": "B4GALNT1", "B4galt1": "B4GALT1", "B4galt2": "B4GALT2",
        "B4galt3": "B4GALT3", "B4galt4": "B4GALT4", "B4galt5": "B4GALT5",
        "B4galt6": "B4GALT6", "B4galt7": "B4GALT7", "Babam1": "BABAM1",
        "Babam2": "BABAM2", "Bace1": "BACE1", "Bace2": "BACE2",
        "Bach1": "BACH1", "Bach2": "BACH2", "Bad": "BAD",
        "Bag1": "BAG1", "Bag2": "BAG2", "Bag3": "BAG3",
        "Bag4": "BAG4", "Bag5": "BAG5", "Bag6": "BAG6",
        "Bai1": "ADGRB1", "Bai2": "ADGRB2", "Bai3": "ADGRB3",
        "Bak1": "BAK1", "Bambi": "BAMBI", "Banf1": "BANF1",
        "Bank1": "BANK1", "Bap1": "BAP1", "Bard1": "BARD1",
        "Basp1": "BASP1", "Batf": "BATF", "Batf2": "BATF2",
        "Batf3": "BATF3", "Bax": "BAX", "Baz1a": "BAZ1A",
        "Baz1b": "BAZ1B", "Baz2a": "BAZ2A", "Baz2b": "BAZ2B",
        "Bbc3": "BBC3", "Bbs1": "BBS1", "Bbs2": "BBS2",
        "Bbs4": "BBS4", "Bbs5": "BBS5", "Bbs7": "BBS7",
        "Bbs9": "BBS9", "Bbx": "BBX", "Bcap31": "BCAP31",
        "Bcar1": "BCAR1", "Bcar3": "BCAR3", "Bcat1": "BCAT1",
        "Bcat2": "BCAT2", "Bccip": "BCCIP", "Bcl10": "BCL10",
        "Bcl11a": "BCL11A", "Bcl11b": "BCL11B", "Bcl2": "BCL2",
        "Bcl2l1": "BCL2L1", "Bcl2l11": "BCL2L11", "Bcl2l2": "BCL2L2",
        "Bcl3": "BCL3", "Bcl6": "BCL6", "Bcl6b": "BCL6B",
        "Bcl7a": "BCL7A", "Bcl7b": "BCL7B", "Bcl7c": "BCL7C",
        "Bcl9": "BCL9", "Bcl9l": "BCL9L", "Bclaf1": "BCLAF1",
        "Bcor": "BCOR", "Bcorl1": "BCORL1", "Bcr": "BCR",
        "Bcs1l": "BCS1L", "Bdnf": "BDNF", "Becn1": "BECN1",
        "Bend3": "BEND3", "Bend4": "BEND4", "Bend5": "BEND5",
        "Best1": "BEST1", "Best2": "BEST2", "Bex3": "BEX3",
        "Bhlhe40": "BHLHE40", "Bhlhe41": "BHLHE41", "Bicd1": "BICD1",
        "Bicd2": "BICD2", "Bid": "BID", "Bik": "BIK",
        "Bin1": "BIN1", "Bin2": "BIN2", "Bin3": "BIN3",
        "Birc2": "BIRC2", "Birc3": "BIRC3", "Birc5": "BIRC5",
        "Birc6": "BIRC6", "Blm": "BLM", "Blmh": "BLMH",
        "Bloc1s1": "BLOC1S1", "Bloc1s2": "BLOC1S2", "Blvra": "BLVRA",
        "Blvrb": "BLVRB", "Bmi1": "BMI1", "Bmp1": "BMP1",
        "Bmp2": "BMP2", "Bmp2k": "BMP2K", "Bmp3": "BMP3",
        "Bmp4": "BMP4", "Bmp5": "BMP5", "Bmp6": "BMP6",
        "Bmp7": "BMP7", "Bmpr1a": "BMPR1A", "Bmpr1b": "BMPR1B",
        "Bmpr2": "BMPR2", "Bmyc": "MYC", "Bnc1": "BNC1",
        "Bnc2": "BNC2", "Bnipl": "BNIPL", "Bod1": "BOD1",
        "Bod1l1": "BOD1L1", "Bok": "BOK", "Bola1": "BOLA1",
        "Bola2": "BOLA2", "Bola3": "BOLA3", "Boll": "BOLL",
        "Bop1": "BOP1", "Bora": "BORA", "Bpgm": "BPGM",
        "Bphl": "BPHL", "Bpi": "BPI", "Bptf": "BPTF",
        "Braf": "BRAF", "Brap": "BRAP", "Brd1": "BRD1",
        "Brd2": "BRD2", "Brd3": "BRD3", "Brd4": "BRD4",
        "Brd7": "BRD7", "Brd8": "BRD8", "Brd9": "BRD9",
        "Brdt": "BRDT", "Bri3": "BRI3", "Bri3bp": "BRI3BP",
        "Bricd5": "BRICD5", "Brip1": "BRIP1", "Brk1": "BRK1",
        "Brms1": "BRMS1", "Brms1l": "BRMS1L", "Brs3": "BRS3",
        "Brs3": "BRS3", "Brs3": "BRS3",
    }
    return curated


def map_to_human_ortholog(gene_symbol, species, ortholog_map):
    """
    Map a mouse or rat gene symbol to human ortholog.
    """
    if species == "Human":
        return gene_symbol.upper()

    # Check curated mapping first
    if gene_symbol in ortholog_map:
        return ortholog_map[gene_symbol]

    # Try uppercase conversion (works for many genes)
    upper = gene_symbol.upper()
    return upper


def parse_gpl1355_annotation():
    """Parse GPL1355 (Affymetrix Rat Genome 230 2.0) annotation file.
    Returns dict: probe_id -> gene_symbol
    """
    gpl_file = os.path.join(DATA_DIR, "GSE61616（7d）", "GPL1355-10794 (1).txt")
    if not os.path.exists(gpl_file):
        logger.warning("GPL1355 annotation file not found: %s", gpl_file)
        return {}

    logger.info("Parsing GPL1355 annotation...")
    probe_to_gene = {}
    with open(gpl_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split("\t")
            if len(parts) < 11:
                continue
            probe_id = parts[0].strip()
            gene_symbol = parts[10].strip()  # Column 11: Gene Symbol
            if gene_symbol and gene_symbol != "---" and gene_symbol != "":
                probe_to_gene[probe_id] = gene_symbol

    logger.info("GPL1355: %d probes mapped to %d unique genes",
                len(probe_to_gene), len(set(probe_to_gene.values())))
    return probe_to_gene


def collapse_probes_to_genes(de_df, probe_to_gene, gene_col="Probe", pval_col="P.Value"):
    """
    Collapse multiple probes to the same gene by taking the most significant probe.
    """
    if gene_col not in de_df.columns:
        logger.error("Column '%s' not found in DE results", gene_col)
        return de_df

    # Map probe IDs to gene symbols
    de_df = de_df.copy()
    de_df["GeneSymbol"] = de_df[gene_col].map(probe_to_gene)

    # Remove probes without gene mapping
    mapped = de_df.dropna(subset=["GeneSymbol"])
    logger.info("Mapped %d/%d probes to genes", len(mapped), len(de_df))

    # For each gene, keep the probe with the lowest p-value
    if pval_col in mapped.columns:
        idx_min_p = mapped.groupby("GeneSymbol")[pval_col].idxmin()
        collapsed = mapped.loc[idx_min_p]
    else:
        # If no p-value column, keep the first occurrence
        collapsed = mapped.drop_duplicates(subset=["GeneSymbol"], keep="first")

    logger.info("Collapsed to %d unique genes", len(collapsed))
    return collapsed


def map_gse104036_to_human():
    """
    GSE104036 uses Mouse gene symbols. Map to human orthologs.
    Uses curated ortholog map + uppercase conversion as fallback.
    """
    de_file = os.path.join(RESULT_DIR, "GSE104036_DE_results.csv")
    if not os.path.exists(de_file):
        logger.warning("GSE104036 DE results not found")
        return None

    logger.info("=" * 50)
    logger.info("Processing GSE104036 (Mouse RNA-seq)")

    de = pd.read_csv(de_file)
    logger.info("Input: %d genes", len(de))

    ortholog_map = get_species_ortholog_map()

    # Map mouse gene symbols to human orthologs
    de["GeneSymbol"] = de["Gene"].apply(
        lambda g: map_to_human_ortholog(g, "Mouse", ortholog_map)
    )

    # Add statistics columns
    out_cols = ["Gene", "GeneSymbol", "logFC", "logCPM", "PValue", "FDR", "Dataset", "Species"]
    out = de[out_cols].copy()
    out = out.rename(columns={"Gene": "OriginalID"})

    out_file = os.path.join(RESULT_DIR, "GSE104036_DE_gene_level.csv")
    out.to_csv(out_file, index=False)
    logger.info("Output: %d genes -> %s", len(out), out_file)

    return out


def process_illumina_de(dataset_name, probe_to_gene_map=None):
    """
    Process Illumina microarray DE results.
    If no probe-to-gene map is provided, use the probe ID as-is (ILMN_xxx).
    """
    de_file = os.path.join(RESULT_DIR, f"{dataset_name}_DE_results.csv")
    if not os.path.exists(de_file):
        logger.warning("%s DE results not found", dataset_name)
        return None

    logger.info("=" * 50)
    logger.info("Processing %s", dataset_name)

    de = pd.read_csv(de_file)
    logger.info("Input: %d probes", len(de))

    if probe_to_gene_map:
        de = collapse_probes_to_genes(de, probe_to_gene_map, gene_col="Probe", pval_col="P.Value")
    else:
        # No mapping available, use probe ID as gene symbol
        de["GeneSymbol"] = de["Probe"]
        logger.warning("No probe-to-gene mapping for %s, using probe IDs as-is", dataset_name)

    out_cols = ["Probe", "GeneSymbol", "logFC", "AveExpr", "P.Value", "adj.P.Val", "Dataset", "Species"]
    available_cols = [c for c in out_cols if c in de.columns]
    out = de[available_cols].copy()
    out = out.rename(columns={"Probe": "OriginalID"})

    out_file = os.path.join(RESULT_DIR, f"{dataset_name}_DE_gene_level.csv")
    out.to_csv(out_file, index=False)
    logger.info("Output: %d genes -> %s", len(out), out_file)

    return out


def process_affy_de(dataset_name, probe_to_gene):
    """Process Affymetrix DE results with probe-to-gene mapping."""
    de_file = os.path.join(RESULT_DIR, f"{dataset_name}_DE_results.csv")
    if not os.path.exists(de_file):
        logger.warning("%s DE results not found", dataset_name)
        return None

    logger.info("=" * 50)
    logger.info("Processing %s (Affymetrix)", dataset_name)

    de = pd.read_csv(de_file)
    logger.info("Input: %d probes", len(de))

    de = collapse_probes_to_genes(de, probe_to_gene, gene_col="Probe", pval_col="P.Value")

    out_cols = ["Probe", "GeneSymbol", "logFC", "AveExpr", "P.Value", "adj.P.Val", "Dataset", "Species"]
    available_cols = [c for c in out_cols if c in de.columns]
    out = de[available_cols].copy()
    out = out.rename(columns={"Probe": "OriginalID"})

    out_file = os.path.join(RESULT_DIR, f"{dataset_name}_DE_gene_level.csv")
    out.to_csv(out_file, index=False)
    logger.info("Output: %d genes -> %s", len(out), out_file)

    return out


def run_gene_level_rra():
    """
    Run Robust Rank Aggregation on gene-level DE results.
    Maps all gene symbols to human orthologs for cross-species integration.
    """
    logger.info("=" * 50)
    logger.info("Running gene-level RRA (with human ortholog mapping)")

    # Load ortholog map
    ortholog_map = get_species_ortholog_map()

    # Collect all gene-level DE results
    gene_de_files = []
    for f in os.listdir(RESULT_DIR):
        if f.endswith("_DE_gene_level.csv"):
            gene_de_files.append(os.path.join(RESULT_DIR, f))

    logger.info("Gene-level DE files: %d", len(gene_de_files))

    if len(gene_de_files) < 2:
        logger.warning("Not enough gene-level DE files for RRA")
        return None

    # Collect all gene symbols and their ranks
    all_genes = set()
    dataset_ranks = {}

    for f in gene_de_files:
        de = pd.read_csv(f)
        ds_name = de["Dataset"].iloc[0] if "Dataset" in de.columns else os.path.basename(f)
        species = de["Species"].iloc[0] if "Species" in de.columns else "Unknown"

        logger.info("Processing %s (Species: %s)", ds_name, species)

        # Determine p-value column (edgeR uses "PValue", limma uses "P.Value")
        pval_col = None
        for candidate in ["P.Value", "PValue", "p.value", "pvalue", "FDR"]:
            if candidate in de.columns:
                pval_col = candidate
                break

        logfc_col = None
        for candidate in ["logFC", "log2FoldChange"]:
            if candidate in de.columns:
                logfc_col = candidate
                break

        if pval_col and logfc_col:
            # Filter out ILMN probe IDs (not real gene symbols)
            valid_genes = de[~de["GeneSymbol"].str.startswith("ILMN_", na=False)].copy()
            logger.info("  %d/%d genes after ILMN filter", len(valid_genes), len(de))

            valid_genes = valid_genes.dropna(subset=[pval_col, logfc_col, "GeneSymbol"])

            # Map to human orthologs
            valid_genes["HumanSymbol"] = valid_genes.apply(
                lambda row: map_to_human_ortholog(row["GeneSymbol"], species, ortholog_map),
                axis=1
            )

            valid_genes["signed_rank"] = -np.log10(
                valid_genes[pval_col].clip(lower=1e-300)
            ) * np.sign(valid_genes[logfc_col])
            valid_genes = valid_genes.sort_values("signed_rank", ascending=False)

            # Keep top entry per human gene symbol
            valid_genes = valid_genes.drop_duplicates(subset=["HumanSymbol"], keep="first")

            ranks = dict(zip(valid_genes["HumanSymbol"], valid_genes["signed_rank"]))
            dataset_ranks[ds_name] = ranks
            all_genes.update(valid_genes["HumanSymbol"])
            logger.info("  %s: %d human genes ranked", ds_name, len(ranks))
        else:
            logger.warning("  %s missing p-value or logFC columns (cols: %s)",
                           ds_name, list(de.columns))

    if len(dataset_ranks) < 2:
        logger.warning("Not enough valid datasets for RRA")
        return None

    # Simple RRA-like aggregation: for each gene, compute median rank across datasets
    logger.info("Total unique human genes across all datasets: %d", len(all_genes))

    rra_results = []
    for gene in all_genes:
        ranks = []
        n_datasets = 0
        for ds_name, ds_ranks in dataset_ranks.items():
            if gene in ds_ranks:
                ranks.append(ds_ranks[gene])
                n_datasets += 1

        if n_datasets >= 2:
            median_rank = np.median(ranks)
            up_count = sum(1 for r in ranks if r > 0)
            down_count = sum(1 for r in ranks if r < 0)
            direction = "Up" if up_count > down_count else "Down"

            rra_results.append({
                "GeneSymbol": gene,
                "MedianRank": median_rank,
                "N_Datasets": n_datasets,
                "Up_Count": up_count,
                "Down_Count": down_count,
                "Direction": direction,
                "AllRanks": ",".join(f"{r:.2f}" for r in ranks),
            })

    rra_df = pd.DataFrame(rra_results)
    rra_df = rra_df.sort_values("MedianRank", ascending=False)

    out_file = os.path.join(RESULT_DIR, "RRA_gene_level_integrated.csv")
    rra_df.to_csv(out_file, index=False)
    logger.info("RRA results: %d human genes -> %s", len(rra_df), out_file)
    logger.info("Top 20 genes:")
    for _, row in rra_df.head(20).iterrows():
        logger.info("  %s: rank=%.2f, n=%d, dir=%s",
                    row["GeneSymbol"], row["MedianRank"], row["N_Datasets"], row["Direction"])

    return rra_df


def main():
    logger.info("=" * 60)
    logger.info("Phase 1: Probe-to-Gene Mapping + Gene-level RRA")
    logger.info("Start time: %s", datetime.now().isoformat())

    # 1. Parse GPL1355 annotation for Affymetrix arrays
    gpl1355_map = parse_gpl1355_annotation()

    # 2. Load ILMN probe-to-gene mapping for Illumina arrays
    ilmn_map = parse_ilmn_annotation()

    # 3. Process GSE104036 (Mouse) - map to human orthologs
    map_gse104036_to_human()

    # 4. Process Illumina datasets (GSE16561, GSE37587) with ILMN mapping
    process_illumina_de("GSE16561", ilmn_map)
    process_illumina_de("GSE37587", ilmn_map)

    # 5. Process Affymetrix datasets with GPL1355 mapping
    process_affy_de("GSE61616", gpl1355_map)
    process_affy_de("GSE97537", gpl1355_map)

    # 6. Run gene-level RRA
    rra_df = run_gene_level_rra()

    # 6. Intersect with ferroaging genes
    ferroaging_file = os.path.join(RESULT_DIR, "ferroaging_genes_96.csv")
    if rra_df is not None and os.path.exists(ferroaging_file):
        logger.info("=" * 50)
        logger.info("Intersecting RRA results with 96 ferroaging genes")

        fa_genes = pd.read_csv(ferroaging_file)["gene_symbol"].tolist()
        rra_fa = rra_df[rra_df["GeneSymbol"].isin(fa_genes)]
        logger.info("Ferroaging genes in RRA results: %d/%d", len(rra_fa), len(fa_genes))

        if len(rra_fa) > 0:
            top_fa = rra_fa.head(20)
            logger.info("Top ferroaging genes by RRA rank:")
            for _, row in top_fa.iterrows():
                logger.info("  %s: rank=%.2f, n=%d, dir=%s",
                            row["GeneSymbol"], row["MedianRank"],
                            row["N_Datasets"], row["Direction"])

            intersection_file = os.path.join(RESULT_DIR, "ferroaging_genes_RRA_intersection.csv")
            rra_fa.to_csv(intersection_file, index=False)
            logger.info("Intersection saved to: %s", intersection_file)

    logger.info("End time: %s", datetime.now().isoformat())
    logger.info("Done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)