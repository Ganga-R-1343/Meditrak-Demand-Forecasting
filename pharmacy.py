import os
import sys
import io
import webbrowser
from datetime import datetime
from threading import Timer

import matplotlib

matplotlib.use('Agg')  # Suppress Matplotlib mainthread warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==============================================================================
# 0. AUTOMATIC BROWSER OPENER FOR LOCAL EXECUTION
# ==============================================================================
def open_browser():
    webbrowser.open_new_tab("http://localhost:8501")


if __name__ == "__main__":
    if not st.runtime.exists():
        from streamlit.web import cli as stcli

        Timer(2.0, open_browser).start()
        sys.argv = [
            "streamlit",
            "run",
            __file__,
            "--browser.gatherUsageStats=false",
            "--server.headless=true"
        ]
        sys.exit(stcli.main())

# ==============================================================================
# 1. RAW DATASET INGESTION & PIPELINE
# ==============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

RAW_PHARMACY_DATA = """inventory_id\tpharmacy_id\tpharmacy_name\tpharmacy_street_address\tpharmacy_city\tpharmacy_state\tpharmacy_postal_code\tpharmacy_country\tproduct_id\tproduct_name\tproduct_category\tproduct_unit
1\t101\tCentral City Pharmacy\t123 Main St\tNew York\tNY\t10001\tUSA\t2001\tAcetaminophen Extra Strength\tAnalgesic\ttablet
2\t102\tRiverside Wellness Rx\t45 Riverside Dr\tChicago\tIL\t60601\tUSA\t2002\tVitamin D3 2000 IU\tSupplement\tbottle
3\t103\tMountainView Rx\t312 Aspen Rd\tBoulder\tCO\t80301\tUSA\t2003\tMetformin Hydrochloride\tAntidiabetic\ttablet
4\t104\tSunrise Drugs\t19 Sunrise Way\tMiami\tFL\t33101\tUSA\t2004\tIbuprofen 400mg\tAnalgesic\ttablet
5\t105\tGlobal Pharmacy\t88 King St\tLondon\tGreater London\tSW1A2AB\tUK\t2005\tOmega-3 Fish Oil\tSupplement\tbox
6\t106\tWellness Pharmacy\t11 Rue de l'Église\tParis\tÎle-de-France\t75008\tFrance\t2006\tMelatonin Sleep Aid\tSleep Aid\tbox
7\t107\tElderCare Pharmacy\t790 Elm St\tPhoenix\tAZ\t85001\tUSA\t2007\tLosartan Potassium\tAntihypertensive\ttablet
8\t108\tMetroMed Pharmacy\t456 Market Blvd\tSan Francisco\tCA\t94103\tUSA\t2008\tLevothyroxine Sodium\tThyroid\ttablet
9\t109\tSuburban Drugs\t22 Maple Ave\tHouston\tTX\t77002\tUSA\t2009\tAtorvastatin Calcium\tLipid-lowering\ttablet
10\t110\tPharmaAsia\t22 Orchard Rd\tSingapore\tCentral\t238895\tSingapore\t2010\tCalcium + Vitamin D\tSupplement\tbottle
11\t111\tBroadway Rx\t98 Broadway\tSeattle\tWA\t98101\tUSA\t2011\tParacetamol 500mg\tAnalgesic\ttablet
12\t112\tHarbor Drugs\t5 Harbor St\tBoston\tMA\t2101\tUSA\t2012\tCetirizine Hydrochloride\tAntihistamine\ttablet
13\t113\tCountrySide Pharmacy\t17 Pine Lane\tDes Moines\tIA\t50309\tUSA\t2013\tHydrocodone Acetaminophen\tOpioid\ttablet
14\t114\tCityLife Drugs\t34 Queen St\tManchester\tGreater Manchester\tM21AB\tUK\t2014\tSimvastatin 20mg\tLipid-lowering\ttablet
15\t115\tNight Owl Pharmacy\t55 Sunset Blvd\tLos Angeles\tCA\t90001\tUSA\t2015\tDiphenhydramine HCl\tSleep Aid\ttablet
16\t116\tSunrise Wellness\t8 Beach Dr\tJacksonville\tFL\t32202\tUSA\t2016\tVitamin B12 Complex\tSupplement\tbox
17\t117\tPeak Health Rx\t67 Ridge Rd\tDenver\tCO\t80203\tUSA\t2017\tPrednisone 10mg\tSteroid\ttablet
18\t118\tGreenLeaf Pharmacy\t202 Oak St\tPortland\tOR\t97201\tUSA\t2018\tAspirin 325mg\tAnalgesic\ttablet
19\t119\tPharmaDirect\t23 Robson St\tVancouver\tBritish Columbia\tV6B2B7\tCanada\t2019\tOmeprazole 20mg\tAntacid\tcapsule
20\t120\tSummit Pharmacy\t9 Alpine Ave\tSalt Lake City\tUT\t84101\tUSA\t2020\tGlipizide ER\tAntidiabetic\ttablet
21\t121\tDowntown Pharmacy\t77 Lincoln Rd\tDetroit\tMI\t48201\tUSA\t2021\tMagnesium Citrate\tSupplement\tbottle
22\t122\tBay Area Rx\t401 Ocean Dr\tSan Francisco\tCA\t94105\tUSA\t2022\tClopidogrel Bisulfate\tAnticoagulant\ttablet
23\t123\tCareFirst Pharmacy\t81 Maple St\tAtlanta\tGA\t30301\tUSA\t2023\tZolpidem Tartrate\tSleep Aid\ttablet
24\t124\tPharmaCentral\t31 Main St\tDallas\tTX\t75201\tUSA\t2024\tRosuvastatin Calcium\tLipid-lowering\ttablet
25\t125\tSunset Drugs\t12 Sunset Blvd\tLos Angeles\tCA\t90002\tUSA\t2025\tVitamin C 1000mg\tSupplement\tbox
26\t126\tRapid Relief Rx\t204 5th Ave\tMinneapolis\tMN\t55401\tUSA\t2026\tAlbuterol Sulfate\tRespiratory\tinhaler
27\t127\tWellSpring Pharmacy\t102 Spring St\tBaltimore\tMD\t21201\tUSA\t2027\tCalcium Magnesium Zinc\tSupplement\tbottle
28\t128\tMetroPharm Rx\t233 7th Ave\tNew York\tNY\t10011\tUSA\t2028\tMontelukast Sodium\tRespiratory\ttablet
29\t129\tHealthy Roots Pharmacy\t88 Oak Ave\tNashville\tTN\t37201\tUSA\t2029\tVitamin E Softgels\tSupplement\tbottle
30\t130\tCityCare Pharmacy\t77 Broadway\tBoston\tMA\t2109\tUSA\t2030\tFluticasone Propionate\tRespiratory\tinhaler
31\t131\tCornerstone Pharmacy\t99 3rd Ave\tSeattle\tWA\t98102\tUSA\t2031\tMeloxicam 15mg\tAnalgesic\ttablet
32\t132\tPharmaNet\t15 Queen St\tToronto\tOntario\tM5H2N2\tCanada\t2032\tGlimepiride 2mg\tAntidiabetic\ttablet
33\t133\tCareRx Pharmacy\t17 King St\tToronto\tOntario\tM5H2N4\tCanada\t2033\tVitamin D3 1000 IU\tSupplement\tbottle
34\t134\tPharmaWorld\t41 Park Ave\tLondon\tGreater London\tSW1A2BX\tUK\t2034\tAmlodipine Besylate\tAntihypertensive\ttablet
35\t135\tSunshine Drugs\t12 Summer St\tSan Diego\tCA\t92101\tUSA\t2035\tCetrizine 10mg\tAntihistamine\ttablet
36\t136\tCarePlus Pharmacy\t21 Elm St\tOrlando\tFL\t32801\tUSA\t2036\tSimvastatin 40mg\tLipid-lowering\ttablet
37\t137\tPharmaExpress\t44 East St\tBirmingham\tWest Midlands\tB1 1AA\tUK\t2037\tFluticasone Furoate\tRespiratory\tinhaler
38\t138\tLifeCare Pharmacy\t77 King St\tLondon\tGreater London\tSW1A2AC\tUK\t2038\tVitamin B Complex\tSupplement\tbox
39\t139\tSunrise Rx\t402 Lakeview Ave\tOrlando\tFL\t32804\tUSA\t2039\tIbuprofen 200mg\tAnalgesic\ttablet
40\t140\tPharmaPlus\t10 Orchard Rd\tSingapore\tCentral\t238896\tSingapore\t2040\tVitamin D3 5000 IU\tSupplement\tbottle
41\t141\tHealthFirst Pharmacy\t88 Park St\tSan Jose\tCA\t95112\tUSA\t2041\tHydroxyzine HCl\tAntihistamine\ttablet
42\t142\tUrban Rx\t12 Liberty St\tBoston\tMA\t2110\tUSA\t2042\tRosuvastatin 10mg\tLipid-lowering\ttablet
43\t143\tHealthHub Pharmacy\t99 Innovation Dr\tSan Jose\tCA\t95134\tUSA\t2043\tVitamin D3 400 IU\tSupplement\tbottle
44\t144\tPharmaHealth\t201 Orchard Rd\tSingapore\tCentral\t238897\tSingapore\t2044\tVitamin C Chewable\tSupplement\tbox
45\t145\tCityRx Pharmacy\t99 Main St\tChicago\tIL\t60602\tUSA\t2045\tLisinopril 10mg\tAntihypertensive\ttablet
46\t146\tRural Pharmacy\t4 Farm Rd\tBoise\tID\t83702\tUSA\t2046\tMorphine Sulfate\tOpioid\ttablet
47\t147\tPharmaGo\t22 Park Rd\tSingapore\tCentral\t238899\tSingapore\t2047\tVitamin D3 2000 IU\tSupplement\tbottle
48\t148\tSunset Rx\t15 Sunrise Ave\tMiami\tFL\t33102\tUSA\t2048\tAcetaminophen 325mg\tAnalgesic\ttablet
49\t149\tGreen Pharmacy\t100 Forest Rd\tSeattle\tWA\t98103\tUSA\t2049\tVitamin D3 5000 IU\tSupplement\tbottle
50\t150\tWellCare Pharmacy\t55 Wellness Rd\tLondon\tGreater London\tSW1A2AD\tUK\t2050\tVitamin B12 Complex\tSupplement\tbox
51\t201\tUrbanCare Pharmacy\t1421 Oakridge Ave\tBoston\tMA\t2110\tUSA\t3051\tAcetaminophen Extra Strength\tAnalgesic\ttablet
52\t202\tWellness Rx\t227 Maple Lane\tHouston\tTX\t77002\tUSA\t3052\tVitamin D3 2000 IU\tSupplement\tbottle
53\t203\tSummit Drugs\t3507 Ridge Rd\tDenver\tCO\t80205\tUSA\t3053\tMetformin Hydrochloride XR\tAntidiabetic\ttablet
54\t204\tCentral Health Pharmacy\t762 Main St\tNew York\tNY\t10001\tUSA\t3054\tIbuprofen Liquid Gel\tAnalgesic\tcapsule
55\t205\tLakeside Pharmacy\t1650 Willow Dr\tSeattle\tWA\t98101\tUSA\t3055\tAtorvastatin Calcium\tLipid-lowering\ttablet
56\t206\tMountainView Rx\t319 Aspen Ct\tFlagstaff\tAZ\t86001\tUSA\t3056\tHydrocodone Acetaminophen\tOpioid\ttablet
57\t207\tGlobal Pharmacy\t112 Rue Lafayette\tParis\tIDF\t75009\tFrance\t3057\tMelatonin Sleep Support\tSleep Aid\ttablet
58\t208\tPharmaAsia\t9 Orchard Rd\tSingapore\tSG\t238863\tSingapore\t3058\tOmega-3 Fish Oil 1000mg\tSupplement\tbottle
59\t209\tNorthside Drugs\t480 Kensington Ave\tChicago\tIL\t60616\tUSA\t3059\tLisinopril 20mg\tAntihypertensive\ttablet
60\t210\tSunrise Pharmacy\t1550 Bayshore Blvd\tSan Francisco\tCA\t94109\tUSA\t3060\tCetirizine HCl 10mg\tAntihistamine\ttablet
61\t211\tHeritage Pharmacy\t2301 Liberty St\tLondon\tENG\tW1F 7AE\tUK\t3061\tLevothyroxine Sodium\tThyroid\ttablet
62\t212\tElderCare Pharmacy\t480 Sunset Blvd\tNaples\tFL\t34102\tUSA\t3062\tCalcium Citrate 500mg\tSupplement\ttablet
63\t213\tPrimeHealth Drugs\t701 Cherry Dr\tAustin\tTX\t78701\tUSA\t3063\tWarfarin Sodium\tAnticoagulant\ttablet
64\t214\tSilverline Rx\t804 Pine St\tPhiladelphia\tPA\t19103\tUSA\t3064\tAspirin Buffered 325mg\tAnalgesic\ttablet
65\t215\tBrightMed Pharmacy\t980 Olive Ave\tSan Diego\tCA\t92101\tUSA\t3065\tRosuvastatin Calcium\tLipid-lowering\ttablet
66\t216\tMetroMed Pharmacy\t2201 5th Ave\tLos Angeles\tCA\t90017\tUSA\t3066\tAlbuterol Inhaler 90mcg\tRespiratory\tinhaler
67\t217\tHealthBridge Rx\t1522 Elm St\tDallas\tTX\t75201\tUSA\t3067\tDiphenhydramine HCl\tAntihistamine\ttablet
68\t218\tWellness Pharmacy\t1583 Fairview St\tBirmingham\tENG\tB1 1AA\tUK\t3068\tMagnesium Glycinate\tSupplement\ttablet
69\t219\tCityPoint Drugs\t400 Queen St\tManchester\tENG\tM1 2HX\tUK\t3069\tSimvastatin 40mg\tLipid-lowering\ttablet
70\t220\tNight Owl Pharmacy\t50 Broadway\tNew York\tNY\t10004\tUSA\t3070\tNaproxen Sodium\tAnalgesic\ttablet
71\t221\tSuburban Drugs\t1880 Lakeview Rd\tOrlando\tFL\t32801\tUSA\t3071\tPrednisone 5mg\tSteroid\ttablet
72\t222\tPrimeRx Pharmacy\t3662 Cedar St\tPhoenix\tAZ\t85004\tUSA\t3072\tParacetamol 500mg\tAnalgesic\ttablet
73\t223\tCareFirst Drugs\t2041 Grand Ave\tMinneapolis\tMN\t55402\tUSA\t3073\tGlipizide ER\tAntidiabetic\ttablet
74\t224\tPharmaDirect Rx\t701 Wellington Rd\tLondon\tENG\tSW1A 1AA\tUK\t3074\tVitamin B12 Methylcobalamin\tSupplement\ttablet
75\t225\tPharmaUK\t28 Piccadilly\tLondon\tENG\tW1J 7DF\tUK\t3075\tAmlodipine Besylate\tAntihypertensive\ttablet
76\t226\tHealthMart Drugs\t3250 Park Blvd\tSan Jose\tCA\t95110\tUSA\t3076\tOmeprazole 20mg\tAntacid\tcapsule
77\t227\tCareWell Pharmacy\t2109 Grove St\tDetroit\tMI\t48226\tUSA\t3077\tInsulin Glargine\tAntidiabetic\tvial
78\t228\tPharmaFrance\t14 Avenue Victor Hugo\tParis\tIDF\t75116\tFrance\t3078\tVitamin C 1000mg\tSupplement\ttablet
79\t229\tRxPlus Drugs\t602 Oxford St\tOxford\tENG\tOX1 3AF\tUK\t3079\tLevothyroxine Sodium\tThyroid\ttablet
80\t230\tWellRx Pharmacy\t15 Somerset Ave\tSingapore\tSG\t238164\tSingapore\t3080\tIron Bisglycinate\tSupplement\ttablet
81\t231\tRiverway Rx\t1823 River Rd\tSacramento\tCA\t95814\tUSA\t3081\tGabapentin 300mg\tAnticonvulsant\tcapsule
82\t232\tPharmaSing Drugs\t12 Raffles Blvd\tSingapore\tSG\t39802\tSingapore\t3082\tVitamin E 400 IU\tSupplement\ttablet
83\t233\tGreenleaf Pharmacy\t4007 Forest Dr\tPortland\tOR\t97205\tUSA\t3083\tSertraline HCl\tAntidepressant\ttablet
84\t234\tSunshine Rx\t901 Ocean Dr\tMiami\tFL\t33139\tUSA\t3084\tVitamin B Complex\tSupplement\ttablet
85\t235\tLiberty Drugs\t\tSt. Louis\tMO\t63101\tUSA\t3085\tLosartan Potassium\tAntihypertensive\ttablet
86\t236\tRxCity Pharmacy\t333 City Place\tLas Vegas\tNV\t89101\tUSA\t3086\tZolpidem Tartrate\tSleep Aid\ttablet
87\t237\tHilltop Rx\t445 Summit Ave\tRaleigh\tNC\t27601\tUSA\t3087\tAspirin Buffered 500mg\tAnalgesic\ttablet
88\t238\tPharmaDirect SG\t20 Bukit Timah Rd\tSingapore\tSG\t229638\tSingapore\t3088\tFolic Acid 400mcg\tSupplement\ttablet
89\t239\tWellBeing Pharmacy\t2120 Elm St\tCharlotte\tNC\t28202\tUSA\t3089\tCitalopram HBr\tAntidepressant\ttablet
90\t240\tRxCare Drugs\t3500 Grand Ave\tMinneapolis\tMN\t55408\tUSA\t3090\tIbuprofen 200mg\tAnalgesic\ttablet
91\t241\tCityRx Drugs\t900 Market St\tSan Francisco\tCA\t94103\tUSA\t3091\tAmitriptyline HCl\tAntidepressant\ttablet
92\t242\tRxPlus USA\t4200 Westheimer Rd\tHouston\tTX\t77027\tUSA\t3092\tDiphenhydramine HCl\tAntihistamine\ttablet
93\t243\tAllied Pharmacy\t2760 State St\tSalt Lake City\tUT\t84115\tUSA\t3093\tHydrochlorothiazide\tDiuretic\ttablet
94\t244\tHealthFirst Rx\t1255 Main St\tChicago\tIL\t60602\tUSA\t3094\tCalcium Carbonate 600mg\tSupplement\ttablet
95\t245\tPharmaDirect USA\t2001 Park Ave\tNew York\tNY\t10022\tUSA\t3095\tLoratadine 10mg\tAntihistamine\ttablet
96\t246\tRxWell Pharmacy\t2280 Broadway\tNew York\tNY\t10024\tUSA\t3096\tOmeprazole 40mg\tAntacid\tcapsule
97\t247\tPharmaDirect FR\t31 Rue Saint Antoine\tParis\tIDF\t75004\tFrance\t3097\tVitamin D3 1000 IU\tSupplement\ttablet
98\t248\tCarePlus Rx\t1101 Maple Ave\tLondon\tENG\tE1 6AN\tUK\t3098\tSimvastatin 20mg\tLipid-lowering\ttablet
99\t249\tGoodLife Rx\t1510 King St\tLondon\tENG\tSW1A 2AA\tUK\t3099\tMagnesium Citrate\tSupplement\ttablet
100\t250\tRxWell Drugs\t1900 Mission St\tSan Francisco\tCA\t94110\tUSA\t3100\tAcetaminophen PM\tAnalgesic\ttablet
101\t201\tMetroMed Rx\t1225 Market St\tSan Francisco\tCA\t94103\tUSA\t3001\tAcetaminophen Extra Strength\tAnalgesic\ttablet
102\t202\tWellness Pharmacy\t411 Cedar Lane\tAustin\tTX\t78705\tUSA\t3002\tVitamin D3 5000 IU\tSupplement\tbottle
103\t203\tHarbor Drugs\t1801 Ocean Ave\tMiami\tFL\t33139\tUSA\t3003\tAlbuterol Sulfate Inhaler\tRespiratory\tinhaler
104\t204\tGlobal Pharmacy\t77 Rue de Rivoli\tParis\tÎle-de-France\t75001\tFrance\t3004\tIbuprofen 400mg\tAnalgesic\ttablet
105\t205\tRuralCare Drugs\t391 Main St\tWestfield\tMA\t1085\tUSA\t3005\tMetformin 500mg\tAntidiabetic\ttablet
106\t206\tElderCare Pharmacy\t256 Lakeview Dr\tNaperville\tIL\t60540\tUSA\t3006\tCoQ10 100mg Softgels\tSupplement\tbox
107\t207\tNight Owl Pharmacy\t88 King St\tSeattle\tWA\t98109\tUSA\t3007\tMelatonin 5mg\tSleep Aid\ttablet
108\t208\tPharmaAsia\t101 North Bridge Rd\tSingapore\tCentral\t179105\tSingapore\t3008\tFish Oil 1000mg\tSupplement\tbottle
109\t209\tMountainView Rx\t523 Pine St\tBoulder\tCO\t80302\tUSA\t3009\tSimvastatin 20mg\tLipid-lowering\ttablet
110\t210\tSuburban Drugs\t1621 Oak Ave\tCincinnati\tOH\t45208\tUSA\t3010\tCetirizine 10mg\tAntihistamine\ttablet
111\t211\tDowntown Pharmacy\t2040 Michigan Ave\tChicago\tIL\t60616\tUSA\t3011\tHydroxychloroquine 200mg\tAntimalarial\ttablet
112\t212\tUrban Rx\t1202 Walnut St\tPhiladelphia\tPA\t19107\tUSA\t3012\tAspirin 81mg\tAnalgesic\ttablet
113\t213\tHealthFirst Pharmacy\t303 Main St\tDallas\tTX\t75201\tUSA\t3013\tLevothyroxine 100mcg\tThyroid\ttablet
114\t214\tSunrise Pharmacy\t789 Broadway\tNew York\tNY\t10003\tUSA\t3014\tMagnesium Citrate\tSupplement\tbottle
115\t215\tBay Area Pharmacy\t2020 Shoreline Dr\tAlameda\tCA\t94501\tUSA\t3015\tPrednisone 20mg\tSteroid\ttablet
116\t216\tHeartland Pharmacy\t9002 Ridge Rd\tTulsa\tOK\t74131\tUSA\t3016\tAtorvastatin 40mg\tLipid-lowering\ttablet
117\t217\tHealthy Living Rx\t1001 Willow Ave\tLondon\tEngland\tW1A 1AA\tUK\t3017\tVitamin B Complex\tSupplement\tbottle
118\t218\tRiverfront Drugs\t301 River St\tPortland\tOR\t97209\tUSA\t3018\tOmeprazole 20mg\tAntacid\ttablet
119\t219\tCentral Drugs\t80 Main St\tManchester\tNH\t3101\tUSA\t3019\tWarfarin 5mg\tAnticoagulant\ttablet
120\t220\tPharmaDirect London\t221B Baker St\tLondon\tEngland\tNW1 6XE\tUK\t3020\tParacetamol 500mg\tAnalgesic\ttablet
121\t221\tSunshine Rx\t505 Orange Ave\tOrlando\tFL\t32801\tUSA\t3021\tFolic Acid 400mcg\tSupplement\ttablet
122\t222\tGreen Valley Pharmacy\t302 Maple St\tDenver\tCO\t80203\tUSA\t3022\tInsulin Glargine\tAntidiabetic\tvial
123\t223\tCityChoice Pharmacy\t88 High St\tBoston\tMA\t2110\tUSA\t3023\tRanitidine 150mg\tAntacid\ttablet
124\t224\tCareWell Drugs\t200 Sutter St\tSan Francisco\tCA\t94109\tUSA\t3024\tMorphine Sulfate\tOpioid\ttablet
125\t225\tRxPlus Pharmacy\t808 Elm St\tHouston\tTX\t77002\tUSA\t3025\tZinc 50mg\tSupplement\ttablet
126\t226\tNew Leaf Pharmacy\t65 Woodside Ave\tBrooklyn\tNY\t11231\tUSA\t3026\tLisinopril 10mg\tAntihypertensive\ttablet
127\t227\tMedLink Pharmacy\t201 South St\tBoston\tMA\t2111\tUSA\t3027\tCalcium 600mg\tSupplement\ttablet
128\t228\tCareRx Singapore\t33 Orchard Rd\tSingapore\tCentral\t238895\tSingapore\t3028\tAcetaminophen 500mg\tAnalgesic\ttablet
129\t229\tHealthHub Pharmacy\t451 Lincoln Ave\tChicago\tIL\t60614\tUSA\t3029\tAmlodipine 5mg\tAntihypertensive\ttablet
130\t230\tRxCare France\t11 Boulevard Saint-Michel\tParis\tÎle-de-France\t75005\tFrance\t3030\tIbuprofen 200mg\tAnalgesic\ttablet
131\t231\tCommunity Drugs\t120 Main St\tColumbus\tOH\t43215\tUSA\t3031\tClopidogrel 75mg\tAnticoagulant\ttablet
132\t232\tBlueSky Drugs\t300 6th Ave\tPittsburgh\tPA\t15222\tUSA\t3032\tVitamin C 1000mg\tSupplement\ttablet
133\t233\tCityMed Rx\t22 Lexington Ave\tNew York\tNY\t10010\tUSA\t3033\tDiphenhydramine 25mg\tAntihistamine\ttablet
134\t234\tRxCare UK\t8 Piccadilly\tLondon\tEngland\tW1J 0DA\tUK\t3034\tAspirin 325mg\tAnalgesic\ttablet
135\t235\tLakeview Pharmacy\t400 Lake Ave\tMadison\tWI\t53703\tUSA\t3035\tGlipizide 5mg\tAntidiabetic\ttablet
136\t236\tWellbeing Rx\t1224 Grand Ave\tSt. Paul\tMN\t55105\tUSA\t3036\tDocusate Sodium 100mg\tSupplement\tcapsule
137\t237\tCareFirst Drugs\t401 King St\tAlexandria\tVA\t22314\tUSA\t3037\tMontelukast 10mg\tRespiratory\ttablet
138\t238\tEastside Pharmacy\t601 1st Ave\tSeattle\tWA\t98104\tUSA\t3038\tNiacin 500mg\tSupplement\ttablet
139\t239\tRxDirect France\t90 Avenue de l'Opéra\tParis\tÎle-de-France\t75002\tFrance\t3039\tTramadol 50mg\tOpioid\ttablet
140\t240\tSunMed Singapore\t22 Raffles Blvd\tSingapore\tCentral\t39805\tSingapore\t3040\tVitamin E 400 IU\tSupplement\tbottle
141\t241\tMain Street Pharmacy\t900 Main St\tRaleigh\tNC\t27601\tUSA\t3041\tHydrochlorothiazide 25mg\tAntihypertensive\ttablet
142\t242\tGoodHealth Rx\t505 2nd Ave\tMinneapolis\tMN\t55401\tUSA\t3042\tVitamin K2 100mcg\tSupplement\tcapsule
143\t243\tMedConnect Rx\t33 5th St\tSan Diego\tCA\t92101\tUSA\t3043\tSertraline 50mg\tAntidepressant\ttablet
144\t244\tWestside Pharmacy\t40 Sunset Blvd\tLos Angeles\tCA\t90028\tUSA\t3044\tVitamin B12 1000mcg\tSupplement\ttablet
145\t245\tHealthyRx UK\t20 Oxford St\tLondon\tEngland\tWC1N 3AX\tUK\t3045\tThyroxine Sodium 25mcg\tThyroid\ttablet
146\t246\tWellCare USA\t99 3rd Ave\tNew York\tNY\t10003\tUSA\t3046\tVitamin D2 1000 IU\tSupplement\ttablet
147\t247\tWellness Rx\t88 Willow Dr\tSan Francisco\tCA\t94110\tUSA\t3047\tVitamin B6 50mg\tSupplement\ttablet
148\t248\tPrimeCare Rx\t7002 Elm St\tDallas\tTX\t75206\tUSA\t3048\tSertraline 100mg\tAntidepressant\ttablet
149\t249\tMetroPharma UK\t12 Regent St\tLondon\tEngland\tSW1Y 4PE\tUK\t3049\tParacetamol 650mg\tAnalgesic\ttablet
150\t250\tRxCentral USA\t121 Main St\tChicago\tIL\t60604\tUSA\t3050\tFurosemide 40mg\tDiuretic\ttablet
151\t201\tDowntown Wellness Pharmacy\t419 Central Ave\tSan Francisco\tCA\t94103\tUSA\t3011\tAcetaminophen Extra Strength\tAnalgesic\ttablet
152\t202\tGreen Valley Drugs\t88 Elm St\tAustin\tTX\t78701\tUSA\t3012\tOmegaPro Fish Oil\tSupplement\tbottle
153\t203\tSunrise Pharmacy\t123 Main St\tSeattle\tWA\t98104\tUSA\t3013\tMetformin Hydrochloride\tAntidiabetic\ttablet
154\t204\tUrban Rx Center\t15 King St\tLondon\tEngland\tWC2N6JN\tUK\t3014\tIbuprofen Rapid Relief\tAnalgesic\ttablet
155\t205\tMaple Leaf Drugs\t22 Oak Ave\tToronto\tOntario\tM5J2N8\tCanada\t3015\tHydrochlorothiazide Tablets\tAntihic\ttablet
156\t206\tElderCare Pharmacy\t77 Retirement Ln\tNaples\tFL\t34102\tUSA\t3016\tDaily Vitamin D3\tSupplement\tbox
157\t207\tGlobalRx Singapore\t12 Orchard Rd\tSingapore\tCentral\t238841\tSingapore\t3017\tMelatonin Sleep Tabs\tSleep Aid\ttablet
158\t208\tLakeview Drugs\t41 Lake St\tMadison\tWI\t53703\tUSA\t3018\tSimvastatin 40mg\tLipid-lowering\ttablet
159\t209\tPharmaAsia Health\t101 Bukit Timah Rd\tSingapore\tWest\t229834\tSingapore\t3019\tNature's Zinc Plus\tSupplement\tbottle
160\t210\tSunset Drugs\t22 Park Blvd\tLos Angeles\tCA\t90025\tUSA\t3020\tCetirizine Allergy Relief\tAntihistamine\ttablet
161\t211\tLexington Rx\t320 Main St\tLexington\tKY\t40508\tUSA\t3021\tAmlodipine Besylate\tAntihypertensive\ttablet
162\t212\tSun Pharma USA\t1015 Market St\tSan Diego\tCA\t92101\tUSA\t3022\tVitamin C Gummies\tSupplement\tbox
163\t213\tMountainView Rx\t18 Aspen St\tBoulder\tCO\t80302\tUSA\t3023\tOxycodone IR\tOpioid\ttablet
164\t214\tRiverfront Pharmacy\t400 River Rd\tCincinnati\tOH\t45202\tUSA\t3024\tLevothyroxine Sodium\tThyroid\ttablet
165\t215\tPinecrest Pharmacy\t55 Pine St\tCharlotte\tNC\t28202\tUSA\t3025\tMagnesium Citrate\tSupplement\tbottle
166\t216\tBayside Drugs\t12 Bayside Blvd\tTampa\tFL\t33602\tUSA\t3026\tFluticasone Nasal Spray\tRespiratory\tbottle
167\t217\tWellness Pharmacy\t88 Health Ave\tDenver\tCO\t80203\tUSA\t3027\tCalcium Plus Chewables\tSupplement\tbox
168\t218\tSuburban Drugs\t200 Elm St\tDetroit\tMI\t48226\tUSA\t3028\tLisinopril Tablets\tAntihypertensive\ttablet
169\t219\tCityCare Pharmacy\t300 City Center Dr\tChicago\tIL\t60606\tUSA\t3029\tZyrtec Allergy Tabs\tAntihistamine\ttablet
170\t220\tPharmacie Paris\t14 Rue du Bac\tParis\tIle-de-France\t75007\tFrance\t3030\tParacetamol 500mg\tAnalgesic\ttablet
171\t221\tHealthSource Rx\t72 First Ave\tBoston\tMA\t2110\tUSA\t3031\tOmeprazole Capsules\tAntacid\tbox
172\t222\tVillage Rx\t98 Maple St\tBurlington\tVT\t5401\tUSA\t3032\tWarfarin Sodium\tAnticoagulant\ttablet
173\t223\tMetroMed Pharmacy\t45 5th Ave\tNew York\tNY\t10011\tUSA\t3033\tFluticasone Propionate\tRespiratory\tbottle
174\t224\tGlobal Pharmacy\t99 Queen St\tToronto\tOntario\tM5H2M5\tCanada\t3034\tVitamin B12 Complex\tSupplement\tbottle
175\t225\tCentral Rx\t700 Center St\tDallas\tTX\t75201\tUSA\t3035\tRosuvastatin Calcium\tLipid-lowering\ttablet
176\t226\tHarbor Drugs\t22 Harbor St\tBoston\tMA\t2114\tUSA\t3036\tAcetaminophen PM\tAnalgesic\ttablet
177\t227\tNight Owl Pharmacy\t401 7th Ave\tNew York\tNY\t10018\tUSA\t3037\tSominex Sleep Aid\tSleep Aid\tbox
178\t228\tHilltop Drugs\t12 Hilltop Rd\tPortland\tOR\t97209\tUSA\t3038\tDexamethasone Tablets\tSteroid\ttablet
179\t229\tCityCenter Pharmacy\t200 Main St\tMiami\tFL\t33131\tUSA\t3039\tFish Oil Max\tSupplement\tbottle
180\t230\tPharmaDirect Paris\t17 Boulevard St Michel\tParis\tIle-de-France\t75005\tFrance\t3040\tIbuprofen 400mg\tAnalgesic\ttablet
181\t231\tHealthPort Rx\t50 Port Rd\tSeattle\tWA\t98101\tUSA\t3041\tLevothyroxine Sodium\tThyroid\ttablet
182\t232\tExpress Drugs\t21 Express Ln\tHouston\tTX\t77002\tUSA\t3042\tVitamin D3 Forte\tSupplement\tbox
183\t233\tPharmaUK London\t17 Regent St\tLondon\tEngland\tSW1Y4PT\tUK\t3043\tAspirin 81mg\tAnalgesic\ttablet
184\t234\tCountrySide Rx\t11 Country Rd\tBoise\tID\t83702\tUSA\t3044\tInsulin Glargine\tAntidiabetic\tbox
185\t235\tPharmaFrance Lyon\t19 Rue de la République\tLyon\tAuvergne-Rhône-Alpes\t69002\tFrance\t3045\tMagnesium Plus\tSupplement\tbottle
186\t236\tSunshine Drugs\t33 Sun St\tLas Vegas\tNV\t89101\tUSA\t3046\tVitamin E Capsules\tSupplement\tbox
187\t237\tRiverCity Drugs\t159 River Dr\tSacramento\tCA\t95814\tUSA\t3047\tGlipizide XL\tAntidiabetic\ttablet
188\t238\tUrbanRx Singapore\t75 Raffles Blvd\tSingapore\tCentral\t39799\tSingapore\t3048\tMelatonin Sleep Tabs\tSleep Aid\ttablet
189\t239\tMetroDrugs UK\t230 Oxford St\tLondon\tEngland\tW1D2LT\tUK\t3049\tAmlodipine Besylate\tAntihypertensive\ttablet
190\t240\tPharmaAsia Rx\t200 Orchard Blvd\tSingapore\tCentral\t238841\tSingapore\t3050\tOmegaPro Fish Oil\tSupplement\tbottle
5001\t301\tMetroCare Pharmacy\t2811 Elmwood Ave\tChicago\tIL\t60616\tUSA\t10001\tParacetamol Extra Strength\tAnalgesic\ttablet"""


def run_ml_pipeline():
    """Ingests raw TSV datasets, performs feature transformation & trains Random Forest Regressor."""
    df = pd.read_csv(io.StringIO(RAW_PHARMACY_DATA), sep='\t')

    # Synthesize realistic historical inventory demand levels (Target Variable)
    np.random.seed(42)
    base_demand = np.where(df['product_category'] == 'Analgesic', 45, 25)
    unit_multiplier = np.where(df['product_unit'] == 'box', 1.3, 1.0)
    df['UnitsSold'] = (base_demand * unit_multiplier + np.random.normal(0, 8, len(df))).clip(min=5).astype(int)

    # Encode Categorical Fields for Machine Learning Model Input
    le_cat = LabelEncoder()
    le_unit = LabelEncoder()
    le_country = LabelEncoder()

    df['category_encoded'] = le_cat.fit_transform(df['product_category'].astype(str))
    df['unit_encoded'] = le_unit.fit_transform(df['product_unit'].astype(str))
    df['country_encoded'] = le_country.fit_transform(df['pharmacy_country'].astype(str))

    # Features selected from dataset columns
    feature_cols = ['pharmacy_id', 'product_id', 'category_encoded', 'unit_encoded', 'country_encoded']
    X = df[feature_cols]
    y = df['UnitsSold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    encoders = {
        "cat": le_cat,
        "unit": le_unit,
        "country": le_country
    }

    return df, model, scaler, encoders, metrics, X_train.shape, X_test.shape


# Run ML Pipeline
df_merged, demand_model, demand_scaler, encoders, model_metrics, train_size, test_size = run_ml_pipeline()

# ==============================================================================
# 2. STREAMLIT APPLICATION DASHBOARD
# ==============================================================================
st.set_page_config(page_title="Meditrak Pharmacy Inventory Dashboard", layout="wide", page_icon="💊")

# SVG Tablet Logo Code
MEDITRAK_TABLET_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 80" width="260" height="70">
  <defs>
    <linearGradient id="tabletGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0E7490" />
      <stop offset="100%" stop-color="#06B6D4" />
    </linearGradient>
  </defs>
  <g transform="translate(10, 8)">
    <circle cx="32" cy="32" r="28" fill="url(#tabletGrad)" />
    <line x1="12" y1="32" x2="52" y2="32" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" />
    <path d="M 32 20 V 44" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.9" />
  </g>
  <text x="80" y="46" font-family="'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="32" fill="#0F172A">Medi<tspan fill="#06B6D4">trak</tspan></text>
  <text x="82" y="62" font-family="'Segoe UI', Roboto, sans-serif" font-weight="600" font-size="10" fill="#64748B" letter-spacing="1.5">DEMAND FORECASTING</text>
</svg>
"""

# Sidebar Layout
with st.sidebar:
    st.image(MEDITRAK_TABLET_LOGO_SVG, use_container_width=False)
    st.markdown("---")
    st.markdown("### 💊 **Meditrak Platform**")
    st.caption("Live Pharmacy & Product Inventory Demand Forecasting System")

# Header Section
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image(MEDITRAK_TABLET_LOGO_SVG, width=220)
with col_title:
    st.title("Meditrak Inventory & Replenishment System")
    st.caption("Global Pharmacy Stock Analytics & Real-Time Product Predictor")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Real-Time Predictor",
    "📋 Inventory Order Targets",
    "📈 Regional & Category Analytics",
    "⚙️ Dataset Inspector & Model Architecture"
])

# TAB 1: Prediction Interface
with tab1:
    st.header("Predict Single Product Inventory Demand")
    st.write("Select pharmacy branch and product details from the loaded dataset:")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        pharmacies = df_merged[['pharmacy_id', 'pharmacy_name']].drop_duplicates()
        pharmacy_choice = st.selectbox(
            "Select Pharmacy Branch",
            pharmacies['pharmacy_id'],
            format_func=lambda x: f"{x} - {pharmacies[pharmacies['pharmacy_id'] == x]['pharmacy_name'].values[0]}"
        )

    with col_b:
        products = df_merged[['product_id', 'product_name']].drop_duplicates()
        product_choice = st.selectbox(
            "Select Product",
            products['product_id'],
            format_func=lambda x: f"{x} - {products[products['product_id'] == x]['product_name'].values[0]}"
        )

    with col_c:
        matched_row = df_merged[
            (df_merged['pharmacy_id'] == pharmacy_choice) & (df_merged['product_id'] == product_choice)]

        default_cat = matched_row['product_category'].values[0] if len(matched_row) > 0 else \
        df_merged['product_category'].iloc[0]
        default_unit = matched_row['product_unit'].values[0] if len(matched_row) > 0 else \
        df_merged['product_unit'].iloc[0]
        default_country = matched_row['pharmacy_country'].values[0] if len(matched_row) > 0 else \
        df_merged['pharmacy_country'].iloc[0]

        category = st.selectbox("Product Category", sorted(df_merged['product_category'].unique()),
                                index=sorted(df_merged['product_category'].unique()).index(default_cat))
        unit = st.selectbox("Packaging Unit", sorted(df_merged['product_unit'].unique()),
                            index=sorted(df_merged['product_unit'].unique()).index(default_unit))
        country = st.selectbox("Country Jurisdiction", sorted(df_merged['pharmacy_country'].unique()),
                               index=sorted(df_merged['pharmacy_country'].unique()).index(default_country))

    if st.button("Calculate Immediate Replenishment Demand", type="primary"):
        c_enc = encoders['cat'].transform([category])[0]
        u_enc = encoders['unit'].transform([unit])[0]
        cnt_enc = encoders['country'].transform([country])[0]

        input_data = [[pharmacy_choice, product_choice, c_enc, u_enc, cnt_enc]]
        scaled_input = demand_scaler.transform(input_data)
        predicted_units = demand_model.predict(scaled_input)[0]

        st.markdown("---")
        st.subheader("💡 Forecast Results")
        res1, res2 = st.columns(2)
        res1.metric("Predicted Units Sold (Daily)", f"{round(predicted_units, 2)} Units")
        res2.metric("Recommended Monthly Target Buffer", f"{int(predicted_units * 30 * 1.2)} Units",
                    delta="Includes 20% Reserve Stock")

# TAB 2: Whole Dataset Inventory Target Table
with tab2:
    st.header("📋 Whole Network Automated Inventory Order Targets")
    st.write("Generates automated monthly stock targets for every record in your pharmacy dataset:")

    forecast_table = df_merged[
        ['inventory_id', 'pharmacy_id', 'pharmacy_name', 'pharmacy_country', 'product_id', 'product_name',
         'product_category', 'product_unit']].copy()

    # Batch predict for whole table
    features = df_merged[['pharmacy_id', 'product_id', 'category_encoded', 'unit_encoded', 'country_encoded']]
    forecast_table['Estimated Daily Demand'] = np.round(demand_model.predict(demand_scaler.transform(features)), 1)
    forecast_table['Monthly Order Target'] = (forecast_table['Estimated Daily Demand'] * 30 * 1.2).astype(int)

    st.dataframe(forecast_table, use_container_width=True)

# TAB 3: Visual Analytics
with tab3:
    st.header("📈 Inventory Distribution Analytics")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribution by Product Category")
        fig1, ax1 = plt.subplots()
        cat_counts = df_merged['product_category'].value_counts()
        sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="mako", ax=ax1)
        ax1.set_xlabel("Record Count")
        st.pyplot(fig1)
        plt.close(fig1)

    with g2:
        st.subheader("Pharmacy Presence by Country")
        fig2, ax2 = plt.subplots()
        country_counts = df_merged['pharmacy_country'].value_counts()
        sns.barplot(x=country_counts.index, y=country_counts.values, palette="crest", ax=ax2)
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
        plt.close(fig2)

# TAB 4: Dataset Inspector
with tab4:
    st.header("⚙️ Ingested Dataset & ML Model Metrics")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Dataset Records", f"{df_merged.shape[0]} Rows")
    d2.metric("Training Set (80%)", f"{train_size[0]} Rows")
    d3.metric("Testing Set (20%)", f"{test_size[0]} Rows")

    st.markdown("---")
    st.subheader("🎯 Model Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Accuracy Score", f"{model_metrics['R2']:.4f}")
    m2.metric("Mean Absolute Error (MAE)", f"{model_metrics['MAE']:.2f} Units")
    m3.metric("Mean Squared Error (MSE)", f"{round(model_metrics['MSE'], 2)}")

    st.markdown("---")
    st.subheader("🗃️ Raw Pharmacy Inventory Table Inspection")
    st.dataframe(df_merged[['inventory_id', 'pharmacy_id', 'pharmacy_name', 'pharmacy_street_address', 'pharmacy_city',
                            'pharmacy_state', 'pharmacy_postal_code', 'pharmacy_country', 'product_id', 'product_name',
                            'product_category', 'product_unit']], use_container_width=True)