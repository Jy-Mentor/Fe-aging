"""
Save SenePy universal senescence signatures to CSV for R integration.
Uses only built-in modules to avoid numpy/pandas DLL issues.
SenePy: Sanborn MA et al., Nature Communications (2025), DOI: 10.1038/s41467-025-57047-7
"""

import os
import sys
import csv
import logging
import traceback
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "save_senepy_signatures.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "senepy")
os.makedirs(OUT_DIR, exist_ok=True)

MOUSE_UNIVERSAL = [
    "Hba-a1", "Hbb-b1", "Gm11428", "Vsig4", "Saa3", "Cxcl13", "Igj", "Hbb-b2",
    "Elane", "Fcna", "Slpi", "Ccl5", "Lilrb4", "Gp49a", "Pf4", "H2-Q8",
    "AA467197", "Kcnj8", "C130026I21Rik", "Lyz1", "Ms4a6d", "Snca", "Cdkn2a",
    "Il1b", "Cybb", "Tnfrsf13c", "Ltb", "Ncf4", "Cxcr3", "Bcl2a1b", "Napsa",
    "Cyp4f18", "Csf2rb2", "Ms4a4b", "Prtn3", "2010001M09Rik", "Serpina3g",
    "H2-Q6", "Bcl2a1d", "Ramp3", "Rgs1", "Lrg1", "Hba-a2", "Gm7609",
    "2200002D01Rik", "Cd3g", "Ltb4r1", "Csprs", "Ms4a6c", "Lcn2", "Fgr",
    "Plscr2", "Mnda", "Ms4a7", "Cfp", "Alas2", "Slfn2", "Lat", "1810033B17Rik",
    "Fpr1", "Gpr18", "Cd72", "Timp4", "Cd5l", "Traf1", "Ccl7", "Myct1",
    "Atp6v0c-ps2", "Penk", "Cebpe", "H2-DMb1", "Nkg7", "Clec4n", "Il6",
    "Ms4a1", "Vwf", "H2-DMb2", "Cd163", "Ecscr", "A430084P05Rik", "Cd7",
    "Ccl3", "Cd40", "Gimap3", "Csf2rb", "Aif1", "1700112E06Rik", "Rorb",
    "Ccl17", "Hp", "C4b", "Emcn", "Ccl9", "Ifi47", "S1pr4", "Zc3h12d",
    "Naip2", "Grrp1", "Ccl8", "Asns", "Lilrb3", "Cd300lg", "Retnlg", "Fcnb",
    "1100001G20Rik", "4930506M07Rik", "Cd2", "Clec12a", "Lrrc25", "Lyve1",
    "Ptprc", "Sh2d2a", "Hdc", "Pilra", "Slamf9", "Tnfrsf4", "Apol9b",
    "Folr2", "Ms4a3", "Blk", "Meis3", "Retnla", "Gimap4", "Amica1", "Ccl6",
    "Clec4a2", "S100a4", "Gm12250", "Faim3", "Rspo1", "Beta-s", "4931408D14Rik",
    "Espn", "Mapk11", "Apoc2", "Mpo", "Cd19", "Cd79b", "Gm4951", "Zbp1",
    "Ccl4", "Ch25h", "Sncg", "Gpihbp1", "Ifit3", "Sftpc", "Cst7", "Adam8",
    "Cd48", "Oas1a", "Ccl11", "Ube2c", "Uchl1", "Zbtb7c", "Serpinb6b",
    "Vtn", "Ctsw", "Tspan32", "Fpr2", "Nfe2", "H2-Oa", "4930572J05Rik",
    "Rnase6", "Glipr1", "Ifi204", "Actc1", "Birc5", "Gimap7", "Snx20",
    "Slc4a1", "Ccnb2", "Mpeg1", "Pilrb2", "Ebi3", "Inhbb", "Gm15987",
    "Calml3", "Dok2", "Cd207", "Cxcr6", "Gimap1", "Ltbp2", "Gypa", "Ccr6",
    "Iigp1", "Slamf7", "Stmn2", "Pstpip1", "Ube2l6", "Cd177", "Tnfrsf9",
    "Camp", "Gm13315", "Pglyrp1", "Chi3l3", "Ccr7", "Bank1", "Upp1",
    "Cdkn2b", "Trem3", "Ccr1", "H2-T10", "Sfpi1", "Phlda3", "Ms4a4c",
    "Zmynd15", "Ifitm6", "Ncf2", "Lyl1", "Wisp2", "Sox18", "G0s2", "Rsad2",
    "Meox1", "Fcgr2b", "Pigz", "Ankrd37", "Arhgap9", "Igsf6", "Clps",
    "Slc11a1", "H2-DMa", "Gm4902", "Casp4", "Car4", "Oasl1", "Nlrc5",
    "Slc13a3", "Rassf4", "Smpdl3b", "Cyp2a4", "Fmnl1", "Chi3l1", "Gpr182",
    "Clec4a3", "Cxcl3", "Ptprcap", "Icam2", "Csrp3", "C1rl", "Fgf9",
    "Slitrk2", "Syngr1", "C5ar1", "Gm8369", "Ifi205", "Fcer2a", "Pgf",
    "Cd3d", "Dpep2", "Cd38", "Gda", "Inhba", "Trim30a", "Epsti1", "Cdc20",
    "Ndufa4l2", "Tmem79", "Slc2a6", "Ebf1", "Spns2", "Oas3", "Vpreb3",
    "Cd209f", "1810021B22Rik", "Pcsk1n", "Cp", "Ablim3", "Gm12504", "Rom1",
    "Slfn4", "Mocs1", "Kdm5d", "Lhfpl2", "Fxyd2", "1190002F15Rik", "Tnf",
    "Dem1", "Zbtb12", "Dtx4", "Olfr613", "Ccl24", "H2-M2", "Gm5069", "Mr1",
    "Cd83", "Fyb", "Abcc9", "Spink3", "Sostdc1", "Nppa", "Stx11", "Acp5",
    "Slc7a13", "Ahsg", "Fabp7", "Tnni3", "Itgb7", "Tmem40", "C1qtnf9",
    "Skap1", "AI662270", "Egfl7", "Pkib", "Hcls1", "Casp12", "Itpr3",
    "Prss2", "Mmp8", "Tubb3", "Oasl2", "Cyp2d12", "1700019D03Rik", "Clec7a",
    "Ccl22", "Ly6c2", "H2-Q2", "Dmkn", "Fam26f", "Ms4a4d", "Was", "Gmfg",
    "Il2rg", "Il7", "Wdfy4", "Colec11", "Prr5", "Sp110", "Cyp2a5", "Cyp4b1",
    "Cd274", "Cd93", "Cd79a", "Selp", "Cpeb1", "Clcf1", "Aqp2", "Comp",
    "Cldn5", "Rgs16", "Serpina3n", "Mgl2", "Ciita", "Spns3", "Padi4",
    "Cd52", "Adamtsl5", "Gimap5", "Nat8", "Flt1", "Fam118a", "Mblac1",
    "Tigit", "Vill", "Rhof", "Mfsd7a", "Dusp4", "Src", "Rnls", "Itgb2",
    "Rab25", "Dusp14", "Card6", "Syk", "Tnnt2", "Arg2", "Mocos", "Rhbdl1",
    "Tusc1", "4930413G21Rik", "Lcp2", "Ngp", "Treml2", "Mapk13", "Ffar2",
    "Cdc42ep5", "Psmb9", "Plac8", "H2-Q10", "Slc22a12", "Adora2b", "Sesn2",
    "Zbtb32", "Reg3g", "Cd200r4", "Capsl", "BC064078", "Agmo", "Slc6a8",
    "Cd8a", "BC021785", "Mx2", "Prrg4", "Ssh1", "4930599N23Rik", "Cyp4a10",
    "2010204K13Rik", "Alyref2", "Crlf2", "Cbr2", "Mylpf", "Sox13", "Aldh3b1",
    "Olfm1", "Hs3st1", "Slc15a3", "Mmp23", "Kctd18", "Npr1", "Htra4",
    "Ociad2", "Abcb1b", "Cxcr5", "Serpinb1a", "Tnfsf13b", "Cyp2d22", "Rac2",
    "Clec1b", "Unc119", "Rasgrp2", "Csf1", "Wnt10a", "Fmo2", "Gata2",
    "Dand5", "Myo1f", "Tbc1d10c", "Ugt1a7c", "Ccdc126", "Hip1", "Trim30d",
    "St8sia4", "Ctu1", "Srpk3", "Clec4a1", "Icos", "Clec1a", "Ltf", "Mmp3",
    "Usp28", "Gpnmb", "C3", "Enc1", "Ceacam16", "Il1r2", "Glipr2", "Casp1",
    "Lat2", "Fam71a", "Pla1a", "Cd300a", "Fcrla", "Mndal", "Aqp3", "Tgtp2",
    "Rgs5", "Tcf15", "Ifi44", "Gpr116", "Dnase1l3", "Zfp213", "Prss16",
    "Usp18", "Gngt2", "Tmigd1", "Rbp7", "Cytip", "S1pr1", "Rtp4", "Miox",
    "Spib", "Snhg11", "Gbp2", "Cela2a", "Cdh5", "Slc39a4", "Alox5ap",
    "Hpse", "Krt84", "Aldh3b2", "L1cam", "Basp1", "Krt36", "Cacna1s",
    "AW112010", "Pdcd1", "Zpbp", "Prkar1b", "Zbtb25", "Kap", "Prss34",
    "Them5", "Cldn4", "Paqr4", "Krt24", "Gbp6", "Resp18", "Slfn10-ps",
    "Nt5e", "Sftpa1", "Zfp784", "D730005E14Rik", "Fam151a", "C2cd4b",
    "Upb1", "S100a8", "Krtdap", "Gm10548", "Upk1a", "Rps19-ps3", "Il2rb",
    "Ptx3", "Palm", "Prickle1", "Agt", "Prkcz", "Chst7", "Pdgfa", "Gprc5a",
    "Ifit1", "Urb1", "Ccdc109b", "Zfp462", "Sla2", "Bok", "H2-Ob", "Pcdh1",
    "Gbp3", "Azgp1", "Gchfr", "Tmem27", "H2-Q7", "Rarres2", "Apobec2",
    "Plac9", "Hirip3", "Niacr1", "Gm9733", "Cd53", "C1ra", "C1qtnf6",
    "Tmem71", "Gm15706", "C130036L24Rik", "Lmo1", "Cfb", "Ager", "Ppbp",
    "Apod", "Ppp1r1a", "Sdcbp2", "Alox12", "Pou5f2", "S100a14", "Slc52a3",
    "Fcgr4", "Slc28a3", "Mpl", "Slc13a1", "Oas1b", "Apoc1", "Nqo1", "Hoga1",
    "Cd247", "Grap", "Chst1", "Spn", "Pdzk1ip1", "Lrrc32", "Cxcr2", "C7",
    "Ms4a6b", "Dok3", "Irf7", "Ctla2a", "0610010O12Rik", "Sprr1a", "Oas1g",
    "Folr4", "Slfn1", "Itgb2l", "Scgb1a1", "Cldn2", "Slco2a1", "Pou2f2",
    "Podxl", "Fermt3", "Ace", "Hoxc4", "Treml4", "Tnfaip8l2", "Hmha1",
    "Dtx1", "Sell", "Slc38a5", "Siglecg", "Cdk1", "Itga4", "6330512M04Rik",
    "Slamf6", "Tmem171", "Sult5a1", "Ankrd22", "Fgl2", "Bcl2l14", "Slc7a4",
    "Chst2", "Oas2", "AF251705", "Alox5", "Hcst", "Tnfrsf18", "Syt7",
    "Zdhhc24", "6030408B16Rik", "Itgam", "Clic5", "Ctsg", "A530032D15Rik",
    "Nfam1", "Ncf1", "Lst1", "Lrrc26", "AB124611", "Mbnl3", "Cd8b1",
    "Osbpl7", "Neurl3", "Tmem51", "Ankrd1", "Cd300lf", "Prkcb", "Enpp6",
    "Slc4a8", "Eya2",
]

HUMAN_UNIVERSAL = [
    "MMP9", "ITM2C", "MYL9", "PTGS1", "DMTN", "AKR1B1", "ATAD2", "CEP78",
    "KDM1B", "CCL3", "CA2", "MOB1B", "MMD", "SPARC", "CD36", "MAN1A1",
    "SYK", "FGL2", "CCL2", "IL32", "GPX3", "CLU", "TTC7A", "C4orf48",
    "SYNJ2", "PPFIBP2", "WARS", "FBXO6", "TGFBRAP1", "ZNF143", "C15orf48",
    "TXNDC5", "LPCAT4", "ZNF384", "TRIM14", "HYOU1", "PARG", "LTF", "TOP2A",
    "CLSPN", "NRP1", "GNG11", "PCSK6", "ANO6", "RAB31", "GPR183", "TTYH3",
    "CYP1B1", "TPCN1", "POLE", "COL1A1", "DPP4", "ALYREF", "CORO1A", "ADAM9",
    "HEATR3", "URB1", "ACP2", "PNPLA6", "ELK3", "EPB41L2", "FAM118B", "CCL5",
    "TMEM184C", "BCOR", "PRKAR1B", "STK39", "LGALS3BP", "MEMO1", "FSTL1",
    "TATDN2", "RNGTT", "ACTA2", "IGFBP6", "FAAP100", "CD83", "CDKN2A",
    "CD24", "NAPRT", "GMPR", "C2orf88", "DNAJC16", "TBC1D2B", "STT3A",
    "CKAP5", "HLA-DRB1", "PHF2", "PLXNC1", "TAP2", "UNKL", "SZT2", "CHI3L1",
    "TERF2", "HLA-DPA1", "CBL", "CASK", "MGLL", "MTMR10", "PRKACB", "PAG1",
    "ERAP2", "CHPF2", "ADI1", "TECPR1", "ALG1", "CD320", "SNX19", "WDR3",
    "KLHL5", "CCL4L2", "TYMS", "ETS1", "EIF5AL1", "WDFY3", "B4GAT1",
    "SEMA4C", "HTRA1", "ANKRD52", "RUBCN", "PKD2", "UGGT1", "COX18",
    "TMEM160", "ANTXR2", "RPS6KC1", "MSH2", "AUP1", "ECHS1", "SLC12A2",
    "TRIB2", "RGS1", "MTRNR2L1", "APOL3", "CDA", "CCDC88C", "SNCA", "ACSF3",
    "TCF3", "SMC2", "MCM9", "NACC1", "XAF1", "SND1-IT1", "BID", "SKIV2L",
    "RANBP9", "KCND3", "BRCA2", "CYP4V2", "SCRN1", "MRI1", "NUP210", "VPS54",
    "R3HCC1L", "ZHX3", "NOMO1", "CHD1L", "KLF7", "HPCAL1", "AFAP1", "SLC31A1",
    "XYLT1", "CDK2", "STYX", "COL3A1", "FBLIM1", "TCIRG1", "CENPF", "STON2",
    "ATAD5", "SBF1", "ATP13A1", "E2F3", "CCL4", "CD14", "PMP22", "TAMM41",
    "CALD1", "ZDHHC8", "CMTM3", "IGFBP7", "ZBTB17", "WDR54", "CDCA7", "ATR",
    "NUDT18", "SIAE", "SACS", "MCTP2", "SLC25A39", "SRC", "PIK3R4", "NBPF1",
    "FAM98B", "NSF", "PTAFR", "ZNF362", "DNAJC11", "SLC15A4", "ABCA2",
    "CSGALNACT2", "MOSPD2", "FBXL4", "PPP1R9B", "DUSP10", "TTC9", "RGL1",
    "TGFBI", "LIMK1", "USP28", "SOCS7", "ECE1", "FAM168A", "CRTC1", "TMED8",
    "PHF8", "IGLC3", "ZNF592", "GABPA", "ARNTL2", "DCBLD1", "RHBDD3", "EXOG",
    "RCBTB2", "DNASE1", "TOR1B", "SLC41A1", "IRAK3", "SLC26A2", "ARRDC3",
    "TK1", "RAB24", "ADAP2", "STYXL1", "TUBGCP5", "TECPR2", "CSF1", "SBNO2",
    "RHBDD1", "ATG2B", "LRG1", "ZNF236", "TFCP2", "GHDC", "SEMA4D", "FGFR1",
    "CDKN3", "PITPNM1", "MTA2", "FOXRED2", "PLXNA1", "ZNF516", "NFE2L3",
    "TREML1", "COG6", "MID1", "THADA", "PIGQ", "BRI3BP", "ATP13A2", "TMEM241",
    "PPP2R1B", "ABHD3", "SMIM3", "ZNF318", "RAB30", "RALB", "ENDOD1", "NBL1",
    "PDLIM7", "MICAL3", "OAS1", "MFAP4", "PLTP", "SLC35F2", "LMNB1",
    "PPP1R12C", "SETD1A", "PLPP3", "OPA1", "SPR", "KPNA2", "COL6A2", "CYBB",
    "NEK9", "RASSF3", "PIP5K1C", "MKI67", "RFTN1", "RARA", "FAM117B",
    "EHBP1L1", "RBM38", "SMG5", "BMP1", "TAF4", "FYCO1", "HIST1H2BG", "MROH1",
    "TEX2", "KDELR3", "KCTD20", "PLCB3", "RGS16", "F13A1", "SAP130", "FAM193A",
    "MSANTD4", "PI4KA", "COL27A1", "MEX3C", "NFASC", "MPP1", "PYCR2", "FBXL20",
    "CFB", "ARSA", "TMEM94", "MCM6", "TRUB1", "CEP120", "ZFYVE27", "ALDH5A1",
    "TFF3", "CHST15", "SPHK1", "IDE", "TUBGCP6", "TMEM147", "PID1", "UBE2C",
    "TRAPPC11", "PPP2R5D", "CHML", "LYSMD2", "NRP2", "SLC43A3", "PPARD",
    "NDC1", "ARRDC4", "STRBP", "PLOD3", "PIKFYVE", "SLC35E2B", "MXD1", "NADK",
    "CPD", "LEMD3", "IFNAR2", "NAGA", "RC3H2", "PRR5", "MFSD12", "MZB1",
    "SGSH", "TTN-AS1", "GNA12", "FKBP15", "PGM2L1", "PSTPIP2", "WDR66",
    "H1F0", "CD3E", "ARHGAP21", "LMNA", "CYP2S1", "RAB12", "TRAC", "PARVB",
    "ZBED4", "YIF1B", "SP110", "VIPAS39", "EZH2", "ZC3H7B", "TPM1", "OLR1",
    "BRCA1", "RFNG", "MCM5", "BHLHE40", "SLC40A1", "DHX57", "CHKB", "SPECC1",
    "BIRC5", "EGR1", "HPS3", "MGAM", "POC1B", "CHIT1", "PCED1A", "SLC36A4",
    "ABCC1", "MTRNR2L12", "NUSAP1", "CDK8", "VKORC1", "MOB3A", "FAM189B",
    "CRTC2", "RIT1", "PARP10", "NPLOC4", "PLPPR2", "GPR155", "TBC1D13",
    "CCNA2", "CLASP2", "INTS2", "NOP9", "ATP10B", "ADGRA3", "ELP5", "PORCN",
    "ERGIC3", "SUFU", "HS3ST1", "SRGN", "OGG1", "UBE2S", "MYD88", "UHRF2",
    "PEX26", "ZNRF2", "MYO1E", "CHST3", "METTL22", "R3HDM1", "IKBKE",
    "HSD17B10", "CTSH", "THOP1", "SLC25A32", "SETD1B", "CAMKK2", "ACY1",
    "CASKIN2", "PIK3C2B", "MAP3K12", "DNASE1L1", "ZWILCH", "PRDM1", "FRK",
    "ACVR1", "MYO1B", "ZER1", "GBP4", "CAPN10", "SLC37A4", "XPOT", "FKBP2",
    "HLA-DPB1", "FAM217B", "PHF6", "ANKRD28", "FANCI", "THRB", "EP400",
    "HNRNPH2", "CNOT6", "NCAPD2", "TRIM23", "TNRC18", "RAI1", "LY75", "PANK4",
    "MICALL2", "STIP1", "EXOC2", "AP1B1", "CKAP2", "ECM1", "LMBR1", "CLPB",
    "SLC25A43", "MDN1", "STAT5B", "FAM193B", "METTL21A", "NOL6", "RARRES2",
    "RBL1", "ABHD10", "TELO2", "POLR2E", "PCK2", "AGPS", "MTRR", "PTGDS",
    "MBD3", "UPP1", "CPSF3", "CAMKMT", "BIVM", "TANGO2", "CASP1", "HCFC1",
    "TADA2B", "VPS39", "CENPU", "STAMBPL1", "C11orf24", "HLA-DRA", "PWP2",
    "MCM3AP", "TNFAIP2", "EDEM3", "ENTPD6", "INTS3", "DHFR", "EXTL3",
    "SLC39A8", "CCND2", "DENND4B", "ARAP1", "L3MBTL2", "TMEM45B", "TBC1D10B",
    "CTSK", "HPS5", "TRIM41", "CLCN5", "MAN2B2", "ERMP1", "SLC23A2", "PHKA2",
    "ZSCAN29", "ZNF506", "FLVCR1", "KCTD2", "NSMF", "SIPA1L2", "QSOX1",
    "ZNF513", "PAGR1", "AXIN1", "MOB3B", "MTOR", "MCPH1", "EXOC6", "GALC",
    "G0S2", "ADAL", "POLB", "ZNF667-AS1", "ZNF668", "FUCA2", "LRRC8D",
    "CAMTA2", "EXOC8", "SEC24A", "ITPK1", "RAB11B", "SARS2", "SEC24D", "PRR12",
    "ENC1", "LRRC8A", "DDHD1", "LRCH1", "PI16", "SERTAD1", "BLMH", "TMEM120B",
    "ARMCX5", "SSH1", "POU2F1", "GPR107", "SATB1", "PLEKHG3", "PVT1", "NPL",
    "BMI1", "LMNB2", "GBP3", "LPCAT1", "NPDC1", "PIK3R2", "ECH1", "COCH",
    "AAGAB", "ADGRL1", "TIMP2", "WDR83", "SLC16A14", "EPG5", "RHOQ", "SCRN3",
    "BMP2K", "HAPLN3", "PPP1R21", "EIF4E3", "PHF12", "BLNK", "TAX1BP3",
    "PIK3CA", "SPATA20", "ACAP1", "CPOX", "SPOPL", "GNS", "C1GALT1C1",
    "KIAA0100", "ARRB1", "PKIA", "ZMIZ2", "MICAL2", "TMEM39A", "HS2ST1",
    "RAVER1", "HAUS6", "ANKRD29", "P2RX4", "TMEM260", "PLOD1", "SMAD7",
    "TRMT44", "KIF3B", "NPC1", "RAB29", "SLC9A1", "CCSAP", "ENTPD7", "UGGT2",
    "RFK", "MLLT1", "SPRED2", "IL7R", "TBC1D1", "MGP", "RAMP1", "SLAMF7",
    "ZDHHC17", "TIMP3", "NCBP1", "TRAF3", "ANKRD22", "ARHGAP17", "CAPN15",
    "SIRPA", "METTL4", "NNMT", "LRP5", "B3GAT2", "OGFOD2", "C4orf46", "ZNF692",
    "PIGK", "MON1B", "IL33", "ARHGEF10L", "VPS50", "GK", "FAM160B2", "MTMR6",
    "NINJ2", "SGPP1", "TUBGCP3", "ODF2", "CEP135", "OGFRL1", "WDR45",
    "AKR1C3", "USP24", "FPGS", "ARFGEF2", "BCL6", "INPP5A", "CENPQ", "DSN1",
    "CXXC5", "PACRGL", "ARHGAP10", "STK11IP", "FAXDC2", "TUBGCP4", "PYGL",
    "PPM1F", "PDE5A", "PROS1", "RRP36", "MAP3K5", "CUL9", "BPNT1", "PYGO2",
    "F2R", "PFKFB3", "C12orf75", "MYO5B", "ST6GALNAC4", "IL6R", "UBTD2",
    "FZD1", "GLE1", "ZMYM3", "MTMR2", "SYNGAP1", "NPAT", "XXYLT1", "MCM2",
    "SENP1", "OAF", "SEMA3C", "AHR", "NPIPA1", "FAM53B", "AP4S1", "AMPD2",
    "TAOK2", "HLA-DQA1", "TAB1", "CIDEB", "TTI1", "RAB23", "ATP11B", "NUDT16",
    "SLC39A11", "ZDHHC18", "UBIAD1", "PTCH1", "SFXN5", "RBM12", "NLRX1",
    "SH2B1", "SNX30", "GALNT7", "CD69", "PFKFB2", "MBTD1", "SLC30A1", "CMTR1",
    "GALM", "TMEM175",
]


def main():
    logger.info("=" * 60)
    logger.info("Saving SenePy universal senescence signatures")
    logger.info("Reference: Sanborn MA et al., Nature Communications (2025)")
    logger.info("DOI: 10.1038/s41467-025-57047-7")
    logger.info("Start time: %s", datetime.now().isoformat())

    mouse_path = os.path.join(OUT_DIR, "senepy_mouse_universal_genes.csv")
    with open(mouse_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["gene_symbol", "source", "reference"])
        for gene in MOUSE_UNIVERSAL:
            writer.writerow([gene, "SenePy_Universal_Mouse", "Sanborn_2025_NatCommun"])
    logger.info("Mouse universal: %d genes -> %s", len(MOUSE_UNIVERSAL), mouse_path)

    human_path = os.path.join(OUT_DIR, "senepy_human_universal_genes.csv")
    with open(human_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["gene_symbol", "source", "reference"])
        for gene in HUMAN_UNIVERSAL:
            writer.writerow([gene, "SenePy_Universal_Human", "Sanborn_2025_NatCommun"])
    logger.info("Human universal: %d genes -> %s", len(HUMAN_UNIVERSAL), human_path)

    key_senescence_markers = ["Cdkn2a", "Cdkn1a", "Il6", "Cxcl13", "Cdkn2b",
                              "Il1b", "Mmp9", "Mmp3", "Tnf", "Igfbp7", "Glb1",
                              "Serpine1", "Ccna2", "Ccnb2", "Birc5", "Mki67",
                              "Top2a", "Cdk1", "Cdkn3", "Ube2c"]
    mouse_markers = [g for g in key_senescence_markers if g in set(MOUSE_UNIVERSAL)]
    human_markers = [g for g in key_senescence_markers if g in set(HUMAN_UNIVERSAL)]
    logger.info("Key senescence markers in mouse universal: %s", mouse_markers)
    logger.info("Key senescence markers in human universal: %s", human_markers)

    logger.info("End time: %s", datetime.now().isoformat())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)