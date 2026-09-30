"""
Report Generator for QuadInvestorAI
With FREE Mistral AI integration (Direct HTTP - No package needed)
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Global variable to store API key
_MISTRAL_API_KEY = None

def set_mistral_api_key(key: str):
    """Set the global Mistral API key"""
    global _MISTRAL_API_KEY
    _MISTRAL_API_KEY = key

def get_mistral_api_key():
    """Get the global Mistral API key"""
    return _MISTRAL_API_KEY

# ============================================
# PSX COMPANY DIRECTORY (Complete)
# ============================================

PSX_COMPANY_DIRECTORY = {
    # Power & Energy
    "HUBC": {"name": "Hub Power Company Limited", "sector": "Power Generation", "desc": "One of Pakistan's largest independent power producers with diversified energy assets including coal, hydropower, and renewable energy projects."},
    "KAPCO": {"name": "Kot Addu Power Company Limited", "sector": "Power Generation", "desc": "Thermal power plant operator providing significant electricity to Pakistan's national grid with high efficiency."},
    "TSPL": {"name": "Tri-Star Power Limited", "sector": "Power Generation", "desc": "Independent power producer operating thermal power plants in Pakistan, contributing electricity to the national grid."},
    "KOHP": {"name": "Kohinoor Power Company Limited", "sector": "Power Generation", "desc": "Independent power producer generating electricity through thermal power plants, supplying energy to the national grid."},
    "FTSM": {"name": "Fauji Transmissions (Pvt) Limited", "sector": "Power Transmission", "desc": "Manufacturer and supplier of power transmission equipment including transformers and switchgears for Pakistan's energy sector."},
    # Oil & Gas
    "OGDC": {"name": "Oil & Gas Development Company Limited", "sector": "Oil & Gas", "desc": "Pakistan's largest exploration and production company for oil and natural gas, contributing significantly to national energy needs."},
    "PPL": {"name": "Pakistan Petroleum Limited", "sector": "Oil & Gas", "desc": "Major E&P company operating the Sui gas field and other exploration blocks across Pakistan."},
    "PSO": {"name": "Pakistan State Oil Company Limited", "sector": "Oil & Gas", "desc": "Pakistan's largest oil marketing company, distributing petroleum products through extensive retail network nationwide."},
    "PKOL": {"name": "Pakistan Oilfields Limited", "sector": "Oil & Gas", "desc": "Oil and gas exploration company with productive fields in Khyber Pakhtunkhwa and Punjab regions."},
    "MARI": {"name": "Mari Petroleum Company Limited", "sector": "Oil & Gas", "desc": "Large gas producer operating the Mari field, one of Pakistan's largest natural gas reservoirs."},
    "ATRL": {"name": "Attock Refinery Limited", "sector": "Oil & Gas", "desc": "One of Pakistan's oldest oil refineries processing crude oil into petroleum products for domestic market."},
    "BYCO": {"name": "Byco Petroleum Pakistan Limited", "sector": "Oil & Gas", "desc": "Pakistan's largest oil refinery by capacity with integrated petroleum marketing business."},
    "APL": {"name": "Attock Petroleum Limited", "sector": "Oil & Gas", "desc": "Oil marketing company distributing petroleum products to retail and commercial customers across Pakistan."},
    "SHEL": {"name": "Shell Pakistan Limited", "sector": "Oil & Gas", "desc": "Major oil marketing company offering fuels, lubricants, and retail services nationwide."},
    # Banking
    "HBL": {"name": "Habib Bank Limited", "sector": "Banking", "desc": "Pakistan's largest private commercial bank with extensive domestic and international branch network."},
    "MCB": {"name": "MCB Bank Limited", "sector": "Banking", "desc": "Leading Pakistani commercial bank known for strong profitability and retail banking services."},
    "UBL": {"name": "United Bank Limited", "sector": "Banking", "desc": "Major commercial bank operating domestically and internationally with diverse financial services."},
    "NBP": {"name": "National Bank of Pakistan", "sector": "Banking", "desc": "Pakistan's largest state-owned bank providing comprehensive banking services nationwide."},
    "ABL": {"name": "Allied Bank Limited", "sector": "Banking", "desc": "Prominent Pakistani commercial bank with wide branch network and digital banking platform."},
    "BAFL": {"name": "Bank Alfalah Limited", "sector": "Banking", "desc": "Large commercial bank offering retail, corporate, and digital banking services across Pakistan."},
    "BAHL": {"name": "Bank Al Habib Limited", "sector": "Banking", "desc": "Well-regarded Pakistani bank known for conservative risk management and steady growth."},
    "MEBL": {"name": "Meezan Bank Limited", "sector": "Islamic Banking", "desc": "Pakistan's largest Islamic bank offering Shariah-compliant financial products and services."},
    # Cement
    "LUCK": {"name": "Lucky Cement Limited", "sector": "Cement", "desc": "One of Pakistan's largest cement manufacturers with significant production capacity and export operations."},
    "DGKC": {"name": "D.G. Khan Cement Company Limited", "sector": "Cement", "desc": "Leading cement producer with strong production capacity serving domestic and export markets."},
    "MLCF": {"name": "Maple Leaf Cement Factory Limited", "sector": "Cement", "desc": "Large cement producer with significant capacity serving Pakistani construction sector."},
    "CHCC": {"name": "Cherat Cement Company Limited", "sector": "Cement", "desc": "Mid-sized cement manufacturer serving northern Pakistan markets."},
    "PIOC": {"name": "Pioneer Cement Limited", "sector": "Cement", "desc": "Cement company focused on quality production for construction sector growth."},
    "ACPL": {"name": "Attock Cement Pakistan Limited", "sector": "Cement", "desc": "Cement manufacturer with operations serving northern and central Pakistan markets."},
    "KOHC": {"name": "Kohat Cement Company Limited", "sector": "Cement", "desc": "Fast-growing cement producer in Pakistan's northern region with modern production facilities."},
    "FCCL": {"name": "Fauji Cement Company Limited", "sector": "Cement", "desc": "Cement producer in Punjab with strong distribution network and military group backing."},
    # Fertilizers
    "FFC": {"name": "Fauji Fertilizer Company Limited", "sector": "Fertilizers", "desc": "Pakistan's largest urea producer, supporting agricultural productivity across the country."},
    "FFBL": {"name": "Fauji Fertilizer Bin Qasim Limited", "sector": "Fertilizers", "desc": "Fertilizer manufacturer producing DAP and other products for agriculture sector."},
    "EFERT": {"name": "Engro Fertilizers Limited", "sector": "Fertilizers", "desc": "Major urea and fertilizer producer under Engro Group, serving farming community."},
    "ENGRO": {"name": "Engro Corporation", "sector": "Conglomerate", "desc": "Pakistan's leading conglomerate with businesses in fertilizers, energy, food, and petrochemicals."},
    # Textiles
    "NML": {"name": "Nishat Mills Limited", "sector": "Textiles", "desc": "One of Pakistan's largest textile conglomerates with integrated value chain operations and export focus."},
    "NCL": {"name": "Nishat Chunian Limited", "sector": "Textiles", "desc": "Integrated textile manufacturer in spinning, weaving, and power generation."},
    "GATM": {"name": "Gul Ahmed Textile Mills Limited", "sector": "Textiles", "desc": "Leading textile group producing fabric, home textiles, and retail fashion for export markets."},
    "ILP": {"name": "Interloop Limited", "sector": "Textiles", "desc": "Pakistan's largest hosiery exporter producing socks and yarn for global brands like Nike and Adidas."},
    "TOWL": {"name": "Toweller Limited", "sector": "Textile", "desc": "Manufacturer and exporter of high-quality terry towels and home textile products to international markets."},
    "ASHT": {"name": "Ashraf Textile Mills Limited", "sector": "Textile", "desc": "Integrated textile manufacturer producing yarn, greige fabric, and finished fabric for domestic and export markets."},
    # Automobiles
    "INDU": {"name": "Indus Motor Company Limited", "sector": "Automobiles", "desc": "Toyota's Pakistan partner producing Corolla, Hilux, and other Toyota models for local market."},
    "PSMC": {"name": "Pak Suzuki Motor Company Limited", "sector": "Automobiles", "desc": "Suzuki's Pakistan operation assembling popular small cars and motorcycles."},
    "HCAR": {"name": "Honda Atlas Cars Pakistan Limited", "sector": "Automobiles", "desc": "Joint venture assembling and selling Honda Civic, City, and BR-V for Pakistan's passenger car market."},
    "SAZEW": {"name": "Sazgar Engineering Works Limited", "sector": "Automobiles", "desc": "Pakistani automaker producing Haval SUVs and three-wheelers, pivoting toward electric vehicles."},
    "DWTM": {"name": "Dewan Motors Limited", "sector": "Automobile", "desc": "Automobile assembly and manufacturing company producing vehicles for Pakistani automotive market."},
    # Chemicals
    "SIBL": {"name": "Sitara Chemical Industries Limited", "sector": "Chemicals", "desc": "Leading chemical manufacturer producing chlorine, caustic soda, hydrochloric acid, and other industrial chemicals."},
    "LOTCHEM": {"name": "Lotte Chemical Pakistan Limited", "sector": "Chemicals", "desc": "Produces purified terephthalic acid (PTA) for Pakistan's polyester and textile industries."},
    "ICI": {"name": "ICI Pakistan Limited", "sector": "Chemicals", "desc": "Diversified chemical company manufacturing soda ash, polyester, and life sciences products."},
    # Technology
    "SYS": {"name": "Systems Limited", "sector": "IT & Technology", "desc": "Pakistan's largest IT services and BPO company with significant international revenues."},
    "AVN": {"name": "Avanceon Limited", "sector": "IT & Automation", "desc": "Industrial automation and IT solutions provider for energy and industrial sectors."},
    "NETSOL": {"name": "NetSol Technologies Limited", "sector": "IT & Fintech", "desc": "Global IT company specializing in leasing and finance management software solutions."},
    "AIRLINK": {"name": "Airlink Communication Limited", "sector": "Technology & Retail", "desc": "Pakistan's leading mobile phone distributor and Samsung authorized partner."},
    # Food & Beverages
    "NESTLE": {"name": "Nestle Pakistan Limited", "sector": "Food & Beverage", "desc": "Pakistan subsidiary of global food giant producing dairy, beverages, and food products."},
    "UNITY": {"name": "Unity Foods Limited", "sector": "Food & Beverage", "desc": "Diversified food company in cooking oil, ghee, and grain products for consumers across Pakistan."},
    "PMRS": {"name": "Premier Sugar Mills & Distillery Limited", "sector": "Sugar", "desc": "Sugar manufacturer and distillery producing refined sugar, ethanol, and industrial alcohol."},
    # Pharmaceuticals
    "GLAXO": {"name": "GlaxoSmithKline Pakistan Limited", "sector": "Pharmaceuticals", "desc": "Multinational pharma company producing vaccines and medicines for Pakistani healthcare market."},
    "ABOT": {"name": "Abbott Laboratories Pakistan Limited", "sector": "Pharmaceuticals", "desc": "Pakistani arm of Abbott producing nutritional products, diagnostics, and pharmaceuticals."},
    "SEARLE": {"name": "The Searle Company Limited", "sector": "Pharmaceuticals", "desc": "Major pharmaceutical company producing prescription and OTC medicines for local market."},
    "IBLHL": {"name": "IBL HealthCare Limited", "sector": "Healthcare", "desc": "Healthcare company involved in pharmaceutical distribution, medical devices, and healthcare services."},
    # Steel & Metals
    "ASTL": {"name": "Aisha Steel Mills Limited", "sector": "Steel & Metals", "desc": "Cold-rolled steel manufacturer supplying automotive and appliance industries."},
    "ISL": {"name": "International Steels Limited", "sector": "Steel & Metals", "desc": "Flat steel producer making cold-rolled and galvanized sheets for industrial use."},
    "ASML": {"name": "Amreli Steels Limited", "sector": "Steel & Metals", "desc": "Producer of steel rebars and wire rods for construction sector use."},
    "MUGHAL": {"name": "Mughal Iron & Steel Industries Limited", "sector": "Steel & Metals", "desc": "Steel manufacturer producing rebars and billets for Pakistan's construction market."},
    "BAPL": {"name": "Bolan Casting Limited", "sector": "Engineering", "desc": "Manufacturer of iron and steel castings for automotive, agricultural, and industrial applications."},
    # Misc
    "SSOM": {"name": "Sajjad Sadiq & Company Limited", "sector": "Textile", "desc": "Textile manufacturing company producing cotton yarn and fabric for export markets."},
    "ITANZ": {"name": "Ittefaq Tanning Industries Limited", "sector": "Leather", "desc": "Leading leather manufacturer producing finished leather for footwear, garments, and upholstery industries."},
    "ARCTM": {"name": "Arctec Industries Limited", "sector": "Engineering", "desc": "Engineering company specializing in manufacturing of industrial equipment and machinery."},
    "PACE": {"name": "Pakistan Cables Limited", "sector": "Cables & Electrical", "desc": "Leading manufacturer of electrical cables, wires, and conductors for power transmission."},
    "BELA": {"name": "Bela Automotive Limited", "sector": "Automotive Parts", "desc": "Manufacturer of automotive parts and components for local automotive industry."},
    "DAWH": {"name": "Dawood Hercules Corporation Limited", "sector": "Conglomerate", "desc": "Holding company with investments in fertilizers, energy, and financial services."},
    "PKGS": {"name": "Packages Limited", "sector": "Packaging", "desc": "Pakistan's largest packaging materials manufacturer producing paper, board, and flexible packaging."},
    "COLG": {"name": "Colgate-Palmolive Pakistan Limited", "sector": "FMCG", "desc": "Pakistani subsidiary producing oral care and hygiene consumer goods."},
    "UNILEVER": {"name": "Unilever Pakistan Foods Limited", "sector": "FMCG", "desc": "Major consumer goods company producing household, personal care, and food products."},
    "PAKT": {"name": "Pakistan Tobacco Company Limited", "sector": "Tobacco", "desc": "Pakistan affiliate of British American Tobacco, manufacturing cigarettes and tobacco products."},
    "SNGP": {"name": "Sui Northern Gas Pipelines Limited", "sector": "Utilities", "desc": "Gas utility company distributing natural gas across northern Pakistan including Punjab and KPK."},
    "SSGC": {"name": "Sui Southern Gas Company Limited", "sector": "Utilities", "desc": "Gas distribution utility serving Sindh and Balochistan provinces."},
    "PSEL": {"name": "Pakistan Services Limited", "sector": "Hospitality", "desc": "Operator of Pearl Continental hotels, Pakistan's premier luxury hotel chain."},
    "PNSC": {"name": "Pakistan National Shipping Corporation", "sector": "Shipping", "desc": "Pakistan's state-owned shipping company operating bulk carriers and oil tankers."},
    "TREET": {"name": "Treet Corporation Limited", "sector": "Manufacturing", "desc": "Diversified manufacturer of blades, batteries, and consumer products with export operations."},
    "PAEL": {"name": "Pakistan Elektron Limited", "sector": "Electronics", "desc": "Maker of PEL appliances including ACs, refrigerators, and power distribution equipment."},
    "LOADS": {"name": "Service Industries Limited", "sector": "Footwear", "desc": "Pakistan's largest footwear manufacturer and exporter, also producing tyres and rubber products."},
    # Add these missing stocks
    "BAFS": {"name": "Bannu Woollen Mills Limited", "sector": "Textile", "desc": "Manufacturer of woollen fabrics, blankets, and worsted yarn for domestic and export markets."},
    "SUHJ": {"name": "Sachal Energy Development (Pvt) Limited", "sector": "Power Generation", "desc": "Independent power producer operating a 47MW gas-fired power plant in Sindh, Pakistan."},
    "BAHL": {"name": "Bank Al Habib Limited", "sector": "Banking", "desc": "Well-regarded Pakistani bank known for conservative risk management and steady growth."},
    "BELA": {"name": "Bela Automotive Limited", "sector": "Automotive Parts", "desc": "Manufacturer of automotive parts and components for local automotive industry."},
    "BIFO": {"name": "Bifo Industries Limited", "sector": "Chemicals", "desc": "Manufacturer of industrial chemicals and chemical products."},
    "BIPL": {"name": "BankIslami Pakistan Limited", "sector": "Islamic Banking", "desc": "Leading Islamic bank offering Shariah-compliant financial services."},
    "BOP": {"name": "Bank of Punjab", "sector": "Banking", "desc": "Provincial government-owned bank serving Punjab and nationwide."},
    "BWHL": {"name": "Balochistan Wheels Limited", "sector": "Engineering", "desc": "Manufacturer of automotive wheels and rims for the automobile industry."},
    "CHAS": {"name": "Chashma Sugar Mills Limited", "sector": "Sugar", "desc": "Sugar manufacturer producing refined sugar and industrial alcohol."},
    "CJPL": {"name": "Crescent Jute Products Limited", "sector": "Textile", "desc": "Manufacturer of jute products, sacks, and packaging materials."},
    "DCL": {"name": "Dewan Cement Limited", "sector": "Cement", "desc": "Cement manufacturer producing high-quality cement for construction industry."},
    "DEL": {"name": "Delawar Industries Limited", "sector": "Engineering", "desc": "Manufacturer of engineering products and industrial equipment."},
    "DFML": {"name": "Dewan Farooque Motors Limited", "sector": "Automobile", "desc": "Automobile manufacturer producing vehicles in Pakistan."},
    "DFSM": {"name": "Dewan Farooque Spinning Mills", "sector": "Textile", "desc": "Spinning mill producing yarn for textile industry."},
    "DNCC": {"name": "Dandot Cement Company Limited", "sector": "Cement", "desc": "Cement manufacturing company serving northern Pakistan."},
    "DWSM": {"name": "Dewan Sugar Mills Limited", "sector": "Sugar", "desc": "Sugar manufacturer producing refined sugar and molasses."},
    "DYNO": {"name": "Dyno Chemicals Limited", "sector": "Chemicals", "desc": "Manufacturer of industrial chemicals and chemical products."},
    "FABL": {"name": "Faysal Bank Limited", "sector": "Islamic Banking", "desc": "Leading Islamic bank offering Shariah-compliant banking services."},
    "FASM": {"name": "Faisal Spinning Mills Limited", "sector": "Textile", "desc": "Textile spinning mill producing high-quality yarn."},
    "FRSM": {"name": "Faran Sugar Mills Limited", "sector": "Sugar", "desc": "Sugar manufacturer producing refined sugar and industrial alcohol."},
    "GADT": {"name": "Gadoon Textile Mills Limited", "sector": "Textile", "desc": "Textile manufacturer producing fabric and garments for export."},
    "GAMON": {"name": "Gammon Pakistan Limited", "sector": "Construction", "desc": "Construction and engineering company involved in infrastructure projects."},
    "GGL": {"name": "Ghani Glass Limited", "sector": "Glass", "desc": "Manufacturer of glass containers, bottles, and glass products."},
    "HAFL": {"name": "Hafiz Textile Mills Limited", "sector": "Textile", "desc": "Textile manufacturer producing yarn and fabric."},
    "HICL": {"name": "Habib Insurance Company Limited", "sector": "Insurance", "desc": "Insurance company providing general insurance services."},
    "HINO": {"name": "Hinopak Motors Limited", "sector": "Automobile", "desc": "Manufacturer and assembler of heavy commercial vehicles and buses."},
    "HWQS": {"name": "Haseeb Waqas Sugar Mills", "sector": "Sugar", "desc": "Sugar manufacturer producing refined sugar and by-products."},
    "ICI": {"name": "ICI Pakistan Limited", "sector": "Chemicals", "desc": "Diversified chemical company manufacturing soda ash and life sciences products."},
    "IGIHL": {"name": "IGI Holdings Limited", "sector": "Insurance", "desc": "Insurance holding company with interests in general and life insurance."},
    "INIL": {"name": "International Industries Limited", "sector": "Steel", "desc": "Manufacturer of steel pipes, tubes, and engineering products."},
    "JLICL": {"name": "Jubilee Life Insurance Company Limited", "sector": "Insurance", "desc": "Leading life insurance provider in Pakistan."},
    "KASL": {"name": "KASB Securities Limited", "sector": "Financial Services", "desc": "Securities brokerage and financial services company."},
    "KOSM": {"name": "Kohinoor Spinning Mills Limited", "sector": "Textile", "desc": "Textile spinning mill producing quality yarn."},
    "MIRKS": {"name": "Mirpurkhas Sugar Mills", "sector": "Sugar", "desc": "Sugar manufacturer producing refined sugar."},
    "MSOT": {"name": "Masood Textile Mills Limited", "sector": "Textile", "desc": "Textile manufacturer producing denim and apparel for export."},
    "NATF": {"name": "National Foods Limited", "sector": "Food", "desc": "Leading food company producing spices, sauces, and ready-to-cook products."},
    "NCPL": {"name": "Nishat Chunian Power Limited", "sector": "Power", "desc": "Independent power producer generating electricity."},
    "NEPL": {"name": "Nishat Power Limited", "sector": "Power", "desc": "Independent power producer operating a thermal power plant."},
    "NOPK": {"name": "Nishat (Chunian) Limited", "sector": "Textile", "desc": "Textile manufacturer with integrated spinning, weaving, and processing."},
    "NRSL": {"name": "Nimir Resins Limited", "sector": "Chemicals", "desc": "Manufacturer of resins and chemical products."},
    "OVIS": {"name": "Ovis Pharmaceuticals", "sector": "Pharmaceuticals", "desc": "Pharmaceutical company manufacturing medicines and healthcare products."},
    "PAKRI": {"name": "Pakistan Reinsurance Company Limited", "sector": "Insurance", "desc": "Reinsurance company providing risk management services."},
    "PICT": {"name": "Pakistan International Container Terminal", "sector": "Transport", "desc": "Container terminal operator at Karachi Port."},
    "PKGP": {"name": "Pakistan Global Power", "sector": "Power", "desc": "Independent power producer generating electricity."},
    "PNPL": {"name": "Pakistan National Shipping Corporation", "sector": "Shipping", "desc": "National shipping company operating cargo vessels."},
    "REDCO": {"name": "Redco Textiles Limited", "sector": "Textile", "desc": "Textile manufacturer producing fabric and garments."},
    "RMPL": {"name": "Rupali Polyester Limited", "sector": "Textile", "desc": "Manufacturer of polyester yarn and fiber."},
    "SAPL": {"name": "Sapphire Textile Mills Limited", "sector": "Textile", "desc": "Leading textile manufacturer with integrated operations."},
    "SBL": {"name": "Sitara Energy Limited", "sector": "Power", "desc": "Independent power producer generating electricity."},
    "SERT": {"name": "Service Textile Mills Limited", "sector": "Textile", "desc": "Textile manufacturer producing yarn and fabric."},
    "SFL": {"name": "Sapphire Fibres Limited", "sector": "Textile", "desc": "Manufacturer of synthetic fibers and yarn."},
    "SGML": {"name": "Sindh Fine Glass Limited", "sector": "Glass", "desc": "Manufacturer of glass bottles and containers."},
    "SHAFI": {"name": "Shafi Gluco Chem", "sector": "Chemicals", "desc": "Manufacturer of industrial chemicals and glucose."},
    "SHCM": {"name": "Shabbir Tiles & Ceramics Limited", "sector": "Ceramics", "desc": "Manufacturer of tiles and ceramic products."},
    "SITC": {"name": "Sitara Textile Industries Limited", "sector": "Textile", "desc": "Textile manufacturer producing quality fabric."},
    "SLL": {"name": "S.S. Oil Mills Limited", "sector": "Food", "desc": "Manufacturer of cooking oil and ghee."},
    "SPL": {"name": "Sitara Peroxide Limited", "sector": "Chemicals", "desc": "Manufacturer of hydrogen peroxide and chemicals."},
    "SPWL": {"name": "Sapphire Wind Power Limited", "sector": "Power", "desc": "Wind power producer generating renewable energy."},
    "STCL": {"name": "Shield Corporation Limited", "sector": "Manufacturing", "desc": "Manufacturer of tyres, tubes, and rubber products."},
    "STJT": {"name": "S.J. Textile Mills Limited", "sector": "Textile", "desc": "Textile mill producing high-quality yarn."},
    "SURC": {"name": "Suraj Cotton Mills Limited", "sector": "Textile", "desc": "Manufacturer of cotton yarn and fabric."},
    "TELE": {"name": "Telecard Limited", "sector": "Telecommunications", "desc": "Telecommunications company offering data and voice services."},
    "THCCL": {"name": "Thal Industries Corporation Limited", "sector": "Conglomerate", "desc": "Diversified industrial group with interests in agriculture and engineering."},
    "TPLL": {"name": "Tri-Pack Films Limited", "sector": "Packaging", "desc": "Manufacturer of flexible packaging films."},
    "TRG": {"name": "TRG Pakistan Limited", "sector": "Technology", "desc": "Business process outsourcing (BPO) company with global operations."},
    "TRIPF": {"name": "Tri-Star Insurance Limited", "sector": "Insurance", "desc": "General insurance company providing coverage."},
    "UML": {"name": "United Medical Limited", "sector": "Healthcare", "desc": "Healthcare and medical services provider."},
    "UNIC": {"name": "Unicap Modaraba", "sector": "Modaraba", "desc": "Modaraba company engaged in Islamic financial services."},
    "URMT": {"name": "United Real Estate Modaraba", "sector": "Modaraba", "desc": "Real estate modaraba company."},
    "WTL": {"name": "WorldCall Telecom Limited", "sector": "Telecommunications", "desc": "Telecommunication and broadband services provider."},
    "ZAHID": {"name": "Zahid Jee Textile Mills Limited", "sector": "Textile", "desc": "Textile manufacturer producing quality fabric."},
    "SHDT": {"name": "Shield Corporation Limited", "sector": "Manufacturing", "desc": "Manufacturer of tyres, tubes, and rubber products for automotive and industrial applications."},
    "TICL": {"name": "The Thal Industries Corporation Limited", "sector": "Conglomerate", "desc": "Diversified industrial group with interests in agriculture, engineering, and food processing."},
    
}


# ============================================
# AUTO-GENERATE MISSING COMPANY INFO
# ============================================

def auto_generate_company_info(symbol: str) -> dict:
    """
    Automatically generate company info for any missing symbol
    """
    symbol_upper = symbol.upper().strip()
    
    # Sector detection based on symbol patterns
    sector_mappings = {
        'Banking': ['HBL', 'MCB', 'UBL', 'NBP', 'ABL', 'BAFL', 'BAHL', 'MEBL', 'BOP', 'FABL', 'BIPL', 'JSBL', 'SILK'],
        'Cement': ['LUCK', 'DGKC', 'MLCF', 'CHCC', 'PIOC', 'ACPL', 'KOHC', 'FCCL', 'BWCL', 'DCL', 'FLYNG'],
        'Fertilizer': ['FFC', 'FFBL', 'EFERT', 'ENGRO', 'FATIMA'],
        'Oil & Gas': ['OGDC', 'PPL', 'PSO', 'APL', 'ATRL', 'BYCO', 'MARI', 'SHEL', 'PRL', 'NRL', 'TPL', 'HASCOL', 'SFL'],
        'Power Generation': ['HUBC', 'KAPCO', 'TSPL', 'KOHP', 'FTSM', 'NCPL', 'GADT', 'SPWL', 'PKGP', 'EPQL', 'NEPL', 'PIBTL'],
        'Textile': ['NML', 'NCL', 'GATM', 'ILP', 'TOWL', 'ASHT', 'SSOM', 'FASM', 'CWSM', 'DMTX', 'GADT', 'ICL', 'KOSM', 'SERT', 'SITC'],
        'Automobile': ['INDU', 'PSMC', 'HCAR', 'SAZEW', 'DWTM', 'GHNI', 'ATBA', 'HINO', 'DFML', 'AGTL'],
        'Chemical': ['SIBL', 'LOTCHEM', 'ICI', 'NRSL', 'DYNO', 'BIFO', 'BERG', 'SPL', 'SURC', 'AGRIC'],
        'Manufacturing': ['SHDT', 'STCL'],
        'Conglomerate': ['TICL', 'THCCL'],
        'Technology': ['SYS', 'AVN', 'NETSOL', 'AIRLINK', 'TRG', 'WTL', 'OCTOPUS', 'TELE', 'ZAHID'],
        'Food & Beverage': ['NESTLE', 'UNITY', 'PMRS', 'SAPL', 'NATF', 'MFFL', 'SHNI', 'QUICE', 'HUMAN'],
        'Pharmaceuticals': ['GLAXO', 'ABOT', 'SEARLE', 'IBLHL', 'FEROZ', 'OVIS', 'AGP', 'BIO'],
        'Steel & Metals': ['ASTL', 'ISL', 'ASML', 'MUGHAL', 'BAPL', 'KASL', 'ASL', 'AKBL', 'INIL'],
        'Engineering': ['ARCTM', 'BWHL', 'HWQS', 'MIRKS', 'DWSM', 'SBL', 'DEL', 'AMBL', 'STJT', 'CHAS', 'ANSM', 'FRSM', 'GGL', 'HAFL', 'HIRL', 'NOPK', 'PICT', 'PNPL', 'REDCO', 'SHCM', 'SLL', 'STCL', 'TPLL'],
        'Cables & Electrical': ['PACE', 'PAEL', 'SEL', 'SIEM', 'PICT'],
        'Leather': ['ITANZ', 'BDIL', 'SFL'],
        'Sugar': ['PMRS', 'SHAFI', 'ALNRS', 'JVDC', 'MSCL', 'SML', 'RMPL', 'TSMF', 'MSOT'],
        'Insurance': ['EFU', 'IGIHL', 'ICL', 'JLICL', 'PAKRI', 'AICL', 'AGICO', 'ATIL', 'CSIL', 'HICL'],
        'Modaraba': ['FECM', 'FEM', 'FFLM', 'FMC', 'FNEL', 'FRSM', 'FZCM', 'JSM', 'NCPM', 'OLPM', 'SINDM', 'TMM', 'TRSM', 'USLM'],
        'Mutual Fund': ['ABLFS', 'AKIF', 'ALIF', 'ARTF', 'ASGF', 'ATIF', 'AWTIF', 'BCIF', 'CJIF', 'CPBF', 'CPSF', 'CYIF', 'DAIF', 'FBIF', 'FIBIF', 'FIF', 'FNBF', 'FPAF', 'FSVL', 'FWF', 'HIF', 'HWEF', 'HWF', 'ICCF', 'ICIF', 'IDIF', 'IICF', 'IMKF', 'ISETF', 'JSIF', 'JSMF', 'JSMIF', 'KSEETF', 'MCBFSF', 'MEF', 'MEPF', 'MHIF', 'MIIF', 'MZSHF', 'NAGIF', 'NBPIF', 'NBPILS', 'NBPISF', 'NBPNF', 'NBPREITF', 'NCDF', 'NICF', 'NIT', 'NITREITF', 'NLPF', 'PBAF', 'PICF', 'PIMEF', 'PISF', 'PMAF', 'PPAF', 'PPIF', 'PRSF', 'PSIF', 'RMIF', 'SIF', 'SIGF', 'SIPF', 'SPIF', 'SWIF', 'TBF', 'TGSF', 'TMF', 'UAMF', 'UAPF', 'UBSF', 'UFP', 'UIF', 'UILF', 'ULIF', 'UMF', 'UMSF', 'UNAF', 'UNBIAF', 'UNF', 'UNIMF', 'UNREITF', 'USF', 'UTIF', 'UWLF', 'WEIF', 'WFSF', 'WIF', 'WIP', 'WISF', 'WSSF', 'ZIEF', 'ZIGF', 'ZIPF', 'ZITF', 'ZJSF', 'ZLPF', 'ZMSF', 'ZVSF'],
        'Real Estate': ['REIT', 'DCR', 'GLA'],
        # Add to sector_mappings dictionary in auto_generate_company_info
        'Textile': ['NML', 'NCL', 'GATM', 'ILP', 'TOWL', 'ASHT', 'SSOM', 'FASM', 'CWSM', 'DMTX', 'GADT', 'ICL', 'KOSM', 'SERT', 'SITC', 'HAFL', 'SURC', 'STJT', 'REDCO', 'BAFS', 'SAPL', 'SFL', 'MSOT'],
        'Power Generation': ['HUBC', 'KAPCO', 'TSPL', 'KOHP', 'FTSM', 'NCPL', 'GADT', 'SPWL', 'PKGP', 'EPQL', 'NEPL', 'PIBTL', 'SUHJ'],
    }
    
    # Detect sector
    sector = "PSX Listed Company"
    for sec, symbols in sector_mappings.items():
        if symbol_upper in symbols:
            sector = sec
            break
    
    # Check for mutual funds (ends with F)
    if sector == "PSX Listed Company" and (symbol_upper.endswith('F') or 'FUND' in symbol_upper or symbol_upper.endswith('IF')):
        sector = "Mutual Fund"
    
    # Check for modaraba (ends with M)
    if sector == "PSX Listed Company" and symbol_upper.endswith('M'):
        sector = "Modaraba"
    
    # Generate company name
    name_map = {
        'WTL': 'WorldCall Telecom Limited',
        'TRG': 'TRG Pakistan Limited',
        'TELE': 'Telecard Limited',
        'DFML': 'Dewan Farooque Motors Limited',
        'DFSM': 'Dewan Farooque Spinning Mills',
        'PACE': 'Pakistan Cables Limited',
        'BOP': 'Bank of Punjab',
        'FABL': 'Faysal Bank Limited',
        'TPL': 'TPL Corporation Limited',
        'HASCOL': 'Hascol Petroleum Limited',
        'SFL': 'Sapphire Fibres Limited',
        'PRL': 'Pakistan Refinery Limited',
        'NRL': 'National Refinery Limited',
        'GHNI': 'Ghandhara Industries Limited',
        'ATBA': 'Atlas Battery Limited',
        'HINO': 'Hinopak Motors Limited',
        'AGTL': 'Agritech Limited',
        'FEROZ': 'Ferozsons Laboratories Limited',
        'OVIS': 'Ovis Pharmaceuticals',
        'AGP': 'AGP Limited',
        'BIO': 'Biological E. Limited',
        'SHDT': 'Shield Corporation Limited',
        'TICL': 'Thal Industries Corporation Limited',
    }
    
    company_name = name_map.get(symbol_upper, f"{symbol_upper} Limited")
    
    # Generate description based on sector
    descriptions = {
        'Banking': f"{company_name} is a commercial bank operating in Pakistan, offering retail, corporate, and digital banking services to customers nationwide.",
        'Cement': f"{company_name} is a cement manufacturer in Pakistan, producing high-quality cement for the construction industry and infrastructure projects.",
        'Fertilizer': f"{company_name} is a fertilizer company serving Pakistan's agricultural sector with essential plant nutrients and crop solutions.",
        'Oil & Gas': f"{company_name} is engaged in Pakistan's energy sector, involved in oil and gas exploration, production, refining, or marketing.",
        'Power Generation': f"{company_name} is an independent power producer (IPP) generating electricity for Pakistan's national grid.",
        'Textile': f"{company_name} is a textile manufacturer producing yarn, fabric, and garments for domestic and international markets.",
        'Automobile': f"{company_name} is involved in automobile assembly, manufacturing, or parts production in Pakistan.",
        'Chemical': f"{company_name} manufactures industrial chemicals for various applications in Pakistan's industrial sector.",
        'Technology': f"{company_name} provides IT services, software solutions, and technology products for businesses in Pakistan.",
        'Pharmaceuticals': f"{company_name} is a pharmaceutical company manufacturing medicines and healthcare products for the Pakistani market.",
        'Steel & Metals': f"{company_name} is a steel manufacturer producing products for Pakistan's construction and industrial sectors.",
        'Cables & Electrical': f"{company_name} manufactures electrical cables, wires, and power transmission equipment.",
        'Leather': f"{company_name} is a leather manufacturer producing finished leather for footwear, garments, and upholstery.",
        'Engineering': f"{company_name} manufactures industrial equipment, machinery, and engineering products.",
        'Sugar': f"{company_name} is a sugar manufacturer and distillery producing refined sugar and industrial alcohol.",
        'Insurance': f"{company_name} provides insurance and risk management services to individuals and businesses in Pakistan.",
        'Modaraba': f"{company_name} is a modaraba company engaged in Islamic financial services and investments.",
        'Mutual Fund': f"{company_name} is a mutual fund that pools investor money to invest in diversified securities.",
        'Utilities': f"{company_name} is a utility company providing essential services like gas distribution.",
        'Hospitality': f"{company_name} operates hotels, resorts, and hospitality services in Pakistan.",
        'Shipping': f"{company_name} is a shipping company operating cargo and transport services.",
        'Manufacturing': f"{company_name} is a diversified manufacturer of consumer and industrial products.",
        'Footwear': f"{company_name} manufactures footwear products for domestic and export markets.",
        'Electronics': f"{company_name} manufactures electronic appliances and consumer electronics.",
        'Packaging': f"{company_name} is a packaging materials manufacturer producing paper and board products.",
        'FMCG': f"{company_name} manufactures consumer goods including personal care and food products.",
        'Tobacco': f"{company_name} is involved in tobacco processing and cigarette manufacturing.",
        'Conglomerate': f"{company_name} is a diversified holding company with investments across multiple sectors.",
        'Automotive Parts': f"{company_name} manufactures automotive components and parts for the auto industry.",
        'Food & Beverage': f"{company_name} produces food products, beverages, and consumer goods for the Pakistani market.",
        'Healthcare': f"{company_name} provides healthcare services, products, or pharmaceutical distribution.",
        'IT & Technology': f"{company_name} provides IT services, software solutions, and technology products.",
        'Technology & Retail': f"{company_name} distributes technology products and consumer electronics.",
    }
    
    desc = descriptions.get(sector, f"{company_name} is listed on the Pakistan Stock Exchange (PSX).")
    
    # Add specific descriptions for known stocks
    specific_descs = {
        'DFML': "Dewan Farooque Motors is an automobile manufacturer producing vehicles in Pakistan.",
        'WTL': "WorldCall Telecom provides telecommunication and broadband services across Pakistan.",
        'TRG': "TRG Pakistan is a business process outsourcing (BPO) company with global operations.",
        'TELE': "Telecard Limited is a telecommunications company offering data and voice services.",
        'SNGP': "Sui Northern Gas Pipelines distributes natural gas to northern Pakistan.",
        'SSGC': "Sui Southern Gas Company distributes natural gas to Sindh and Balochistan.",
        'BOP': "Bank of Punjab is a provincial government-owned bank serving Punjab and nationwide.",
        'FABL': "Faysal Bank is a leading Islamic bank offering Shariah-compliant banking services.",
        'TPL': "TPL Corporation is a diversified holding company with investments in logistics and insurance.",
        'HASCOL': "Hascol Petroleum is a leading oil marketing company in Pakistan.",
        'PRL': "Pakistan Refinery processes crude oil into petroleum products for domestic consumption.",
        'NRL': "National Refinery produces lube oil, petroleum products, and specialty chemicals.",
    }
    
    if symbol_upper in specific_descs:
        desc = specific_descs[symbol_upper]
    
    return {
        "name": company_name,
        "sector": sector,
        "desc": desc
    }


def get_company_info(symbol: str) -> dict:
    """Return company info from directory; auto-generate if not found"""
    symbol_upper = symbol.upper().strip()
    if symbol_upper in PSX_COMPANY_DIRECTORY:
        return PSX_COMPANY_DIRECTORY[symbol_upper]
    # Auto-generate missing info
    return auto_generate_company_info(symbol_upper)


def get_company_description_via_ai(symbol: str, api_key: str = None) -> str:
    """Get company description using Mistral AI for unknown tickers."""
    info = get_company_info(symbol)
    if info.get("desc") and info["name"] != symbol:
        return f"**{info['name']}** ({info['sector']})\n{info['desc']}"
    key = api_key or get_mistral_api_key()
    if not key:
        return f"**{symbol}** — PSX Listed Company."
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": "You are a financial data assistant. Respond ONLY in JSON with keys: name, sector, desc (1-2 sentences max)."},
                {"role": "user", "content": f"Describe PSX listed company ticker '{symbol}'. JSON only."}
            ],
            "temperature": 0.3,
            "max_tokens": 120,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = json.loads(raw)
            return f"**{parsed.get('name', symbol)}** ({parsed.get('sector', 'PSX')})\n{parsed.get('desc', '')}"
    except Exception:
        pass
    return f"**{symbol}** — PSX Listed Company."


# ============================================
# PLAIN ENGLISH TERMINOLOGY TRANSLATOR
# ============================================

class PlainEnglishTranslator:
    """Translates technical stock terms into simple English"""

    TRANSLATIONS = {
        "uptrend_probability": {"simple": "confidence score", "meaning": "How confident our AI is that this stock will go up"},
        "rsi_14": {
            "simple": "momentum meter",
            "meaning": "Shows if the stock is moving up too fast",
            "interpret": lambda x: "Moving up strongly" if x > 70 else "Moving down" if x < 30 else "Moving normally",
        },
        "macd_hist": {
            "simple": "trend strength",
            "meaning": "How strong the current price trend is",
            "interpret": lambda x: "Strong upward trend" if x > 0.5 else "Weak trend",
        },
        "return_20d": {
            "simple": "recent performance",
            "meaning": "How much the stock gained or lost in the last month",
            "interpret": lambda x: f"Up {x*100:.1f}% in last month" if x > 0 else f"Down {abs(x)*100:.1f}% in last month",
        },
        "volume_ratio": {
            "simple": "trading activity",
            "meaning": "More trading than usual means more interest",
            "interpret": lambda x: "Very active - lots of interest" if x > 1.5 else "Normal activity",
        },
        "close": {"simple": "current price", "meaning": "Latest trading price of the stock"},
        "volatility_10": {"simple": "price stability", "meaning": "How much the price jumps up and down"},
        "sma_20": {"simple": "average price (last month)", "meaning": "Average closing price over last 20 days"},
    }

    @classmethod
    def explain(cls, term: str, value: float) -> str:
        if term not in cls.TRANSLATIONS:
            return f"• {term.replace('_', ' ').lower()}: {value:.2f}"
        info = cls.TRANSLATIONS[term]
        if "interpret" in info:
            return f"• {info['simple']}: {info['interpret'](value)}"
        return f"• {info['simple']}: {value:.2f}"

    @classmethod
    def get_simple_rating(cls, probability: float) -> Dict:
        if probability >= 0.70:
            return {"rating": "Strong Buy", "emoji": "✅", "advice": "Strong potential — multiple technical indicators align for an upward move"}
        elif probability >= 0.60:
            return {"rating": "Buy", "emoji": "📈", "advice": "Positive signals across momentum, volume, and trend indicators"}
        elif probability >= 0.55:
            return {"rating": "Consider", "emoji": "🤔", "advice": "Mixed signals but overall potential exists — monitor closely"}
        elif probability >= 0.50:
            return {"rating": "Watch", "emoji": "👀", "advice": "Wait for clearer signals — not ideal entry timing yet"}
        else:
            return {"rating": "Avoid", "emoji": "⚠️", "advice": "AI does not see strong upside potential at this time"}


# ============================================
# STOCK INSIGHT GENERATOR
# ============================================

def get_stock_insight(symbol: str, metrics: dict) -> str:
    insights = []
    returns = metrics.get('return_20d', 0)
    if returns > 0.05:
        insights.append(f"✅ {symbol} has been performing well recently, up {returns:.1%} in the last month")
    elif returns > 0:
        insights.append(f"📈 {symbol} is showing modest gains of {returns:.1%}")
    elif returns > -0.05:
        insights.append(f"➡️ {symbol} has been stable recently")
    else:
        insights.append(f"⚠️ {symbol} has declined {abs(returns):.1%} in the last month")
    rsi = metrics.get('rsi_14', 50)
    if rsi > 70:
        insights.append(f"⚡ {symbol} has strong upward momentum")
    elif rsi < 30:
        insights.append(f"🔻 {symbol} has been oversold - could be a buying opportunity")
    else:
        insights.append(f"📊 {symbol} has normal momentum")
    volume_ratio = metrics.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        insights.append(f"🔥 Trading activity is {volume_ratio:.1f}x normal - lots of interest")
    return "\n".join(insights)


# ============================================
# PORTFOLIO SUMMARY GENERATOR
# ============================================

class PortfolioSummaryGenerator:
    @staticmethod
    def generate_summary(plan_df, risk_profile: str, total_investment: float) -> str:
        if hasattr(plan_df, 'iterrows'):
            stocks = plan_df[plan_df['symbol'] != 'CASH'] if 'symbol' in plan_df.columns else plan_df
            cash_row = plan_df[plan_df['symbol'] == 'CASH'] if 'symbol' in plan_df.columns else []
            cash = cash_row['actual_investment_pkr'].sum() if len(cash_row) > 0 else 0
        else:
            stocks = []
            cash = 0
        total_invested = total_investment - cash
        invested_percent = (total_invested / total_investment) * 100 if total_investment > 0 else 0
        stock_list = []
        if hasattr(stocks, 'iterrows'):
            for _, row in stocks.iterrows():
                symbol = row.get('symbol', 'Unknown')
                amount = row.get('actual_investment_pkr', 0)
                percent = row.get('allocation_percent', 0)
                prob = row.get('uptrend_probability', 0.5)
                signal = "🎯 High confidence" if prob >= 0.65 else "📈 Good signal" if prob >= 0.55 else "🤔 Mixed signals"
                stock_list.append(f"  • {symbol}: PKR {amount:,.0f} ({percent:.1f}%) - {signal}")
        stock_text = "\n".join(stock_list) if stock_list else "  • No stocks recommended at this time"
        risk_descriptions = {
            "Conservative": "focusing on stability and capital preservation",
            "Moderate": "balancing growth potential with reasonable risk",
            "Aggressive": "seeking higher growth with higher risk tolerance",
        }
        risk_text = risk_descriptions.get(risk_profile, "balanced approach")
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    YOUR INVESTMENT PLAN                      ║
╚══════════════════════════════════════════════════════════════╝

💰 TOTAL INVESTMENT: PKR {total_investment:,.0f}
📊 RISK PROFILE: {risk_profile.upper()} - {risk_text}

📈 WHERE YOUR MONEY GOES:
{stock_text}

💵 UNINVESTED CASH: PKR {cash:,.0f} ({100 - invested_percent:.1f}%)

⚠️ REMEMBER: This is AI-generated guidance, not guaranteed returns.

---
QuadInvestorAI - Making AI investing simple
"""
        return summary.strip()


# ============================================
# STYLED REPORT GENERATOR WITH BOLD HEADINGS & BULLET POINTS
# ============================================

class LLMReportGenerator:
    """Generates DETAILED investment reports using Mistral AI API with styled formatting"""

    def __init__(self, api_key: str = None, use_openai: bool = True):
        self.api_key = api_key or get_mistral_api_key() or os.environ.get("MISTRAL_API_KEY")
        self.use_mistral = self.api_key is not None and len(str(self.api_key)) > 10
        if self.use_mistral:
            print("✅ Mistral AI initialized")
        else:
            print("ℹ️ No API key found. Using template reports.")

    def generate_simple_report(
        self,
        symbol: str,
        probability: float,
        allocation_percent: float,
        amount_pkr: float,
        latest_price: float,
        shares: int,
        target_return_percent: float = 15.0,
        target_profit_pkr: float = None,
        risk_level: str = "Moderate",
        factors: List[dict] = None,
    ) -> str:
        rating = PlainEnglishTranslator.get_simple_rating(probability)
        if target_profit_pkr is None:
            target_profit_pkr = amount_pkr * (target_return_percent / 100)

        # Build readable factor list with bullet points
        factor_lines = []
        if factors:
            for f in factors[:5]:
                feature = f.get('feature', '')
                value = f.get('actual_value', 0)
                direction = f.get('direction', 'neutral')
                factor_lines.append(
                    PlainEnglishTranslator.explain(feature, value) + f" [{direction}]"
                )
        factor_text = "\n".join(factor_lines) if factor_lines else \
            "• AI has analyzed recent price movements and trading patterns"

        # Company context
        company_info = get_company_info(symbol)
        company_name = company_info.get("name", symbol)
        sector = company_info.get("sector", "PSX Listed")
        company_desc = company_info.get("desc", "")

        if self.use_mistral:
            try:
                return self._generate_with_mistral_http(
                    symbol, company_name, sector, company_desc,
                    probability, rating, allocation_percent,
                    amount_pkr, latest_price, shares, target_profit_pkr, factor_text
                )
            except Exception as e:
                print(f"Mistral API error: {e}")

        return self._generate_template(
            symbol, company_name, sector, company_desc,
            probability, rating, allocation_percent,
            amount_pkr, latest_price, shares, target_profit_pkr, factor_text
        )

    def _generate_with_mistral_http(
        self, symbol, company_name, sector, company_desc,
        probability, rating, alloc_percent,
        amount, price, shares, target_profit, factor_text
    ):
        prompt = f"""You are a professional investment analyst writing a detailed stock analysis report for a Pakistani retail investor using the QuadInvestorAI platform.

STOCK DATA:
- Ticker: {symbol}
- Company: {company_name}
- Sector: {sector}
- Background: {company_desc}

AI MODEL OUTPUT:
- AI Confidence Score: {probability:.1%}
- Signal: {rating['rating']} {rating['emoji']}
- Portfolio Allocation: {alloc_percent:.1f}% of capital
- Action: Buy {shares} shares at PKR {price:,.2f} each
- Total Investment: PKR {amount:,.0f}
- Target Profit at 15% return: PKR {target_profit:,.0f}
- Estimated Stop-Loss Level: PKR {price * 0.88:,.2f} (approx -12%)

KEY TECHNICAL SIGNALS DETECTED BY AI:
{factor_text}

Write a DETAILED, STRUCTURED investment report using the six sections below.
IMPORTANT FORMATTING RULES:
1. Make each section heading BOLD by wrapping it in **double asterisks** like: **SECTION 1 - COMPANY OVERVIEW:**
2. Within each section, use bullet points (starting with • or -) to break up information instead of long paragraphs
3. Each bullet point should be on a new line
4. Keep sentences clear and scannable

MINIMUM REQUIREMENTS:
- Total length: 250+ words
- Each section: at least 3 bullet points
- Be specific, informative, and professional

**SECTION 1 - COMPANY OVERVIEW:**
[Write 3-5 bullet points explaining what {company_name} does, the {sector} sector in Pakistan, company's market position, and why it matters to investors]

**SECTION 2 - WHY AI RECOMMENDS THIS STOCK:**
[Write 3-5 bullet points explaining the AI patterns detected. Translate technical signals into plain English. Explain why momentum, volume, and trend data suggest upward movement]

**SECTION 3 - INVESTMENT PLAN:**
[Write 3-5 bullet points with clear action steps including: number of shares to buy, total investment amount, target profit amount, stop-loss level, and recommended holding period]

**SECTION 4 - WHAT TO WATCH:**
[Write 3-5 bullet points of specific factors the investor should monitor: price levels, company news, sector events, macro indicators]

**SECTION 5 - RISK WARNING:**
[Write 3-5 bullet points discussing specific risks for this company and sector in Pakistan, including macro risks]

**SECTION 6 - BOTTOM LINE:**
[Write 2-3 bullet points with a clear, strong conclusion telling the investor exactly what to do]

Write the full report now using the format above:"""

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "mistral-small-latest",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior equity research analyst specialising in the Pakistan Stock Exchange (PSX). "
                        "You write detailed, structured investment reports for retail investors. "
                        "CRITICAL FORMATTING: Use **bold** for section headings (e.g., **SECTION 1 - COMPANY OVERVIEW:**). "
                        "Use bullet points (starting with •) within each section instead of long paragraphs. "
                        "Make the report scannable and easy to read. Be specific, informative, and professional."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.65,
            "max_tokens": 950,
        }
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        raise Exception(f"Mistral API returned {response.status_code}: {response.text[:200]}")

    def _generate_template(
        self, symbol, company_name, sector, company_desc,
        probability, rating, alloc_percent,
        amount, price, shares, target_profit, factor_text
    ):
        return f"""**{rating['emoji']} {symbol} — {rating['rating']}**
{'=' * 60}

**SECTION 1 - COMPANY OVERVIEW:**
• {company_name} ({symbol}) is listed on the Pakistan Stock Exchange and operates in the {sector} sector
• {company_desc if company_desc else f"{symbol} is an established PSX-listed company with significant market presence"}
• The {sector} sector is a key driver of Pakistan's economy, contributing substantially to GDP and employment
• Understanding the business behind the ticker helps investors make more informed decisions about long-term holding potential
• This company operates in a competitive landscape with both domestic and international players

**SECTION 2 - WHY AI RECOMMENDS THIS STOCK:**
• Our ensemble AI model — combining XGBoost pattern recognition and LSTM sequence analysis — assigns {symbol} a confidence score of {probability:.1%}
• This places {symbol} in the {rating['rating']} category, indicating favorable technical conditions
• The model detected the following key signals in recent market data:
{factor_text}
• These signals collectively indicate technical conditions that historically precede upward price movement
• When multiple indicators align in the same direction, the probability of a successful trade increases significantly

**SECTION 3 - INVESTMENT PLAN:**
• **Action:** Buy {shares:,} shares of {symbol} at PKR {price:,.2f} per share
• **Capital required:** PKR {amount:,.0f} (represents {alloc_percent:.1f}% of your portfolio)
• **Target profit:** At 15% gain, profit would be PKR {target_profit:,.0f}, position value becomes PKR {amount + target_profit:,.0f}
• **Stop-loss level:** Set at approximately PKR {price * 0.88:,.2f} (-12%) to limit downside risk
• **Holding period:** Recommended 1-3 months, reassessing after any significant market news

**SECTION 4 - WHAT TO WATCH:**
• **Price action:** Monitor whether {symbol} holds above PKR {price * 0.95:,.2f} in the first two weeks
• **Company news:** Watch for earnings releases, dividend announcements, or major contract wins from {company_name}
• **Sector developments:** Keep track of regulatory changes, industry trends, and competitor movements in the {sector} sector
• **Market direction:** Monitor the KSE-100 index as broad market moves influence individual stock performance
• **Volume pattern:** Increasing trading volume on up days confirms institutional interest

**SECTION 5 - RISK WARNING:**
• **Market risk:** Pakistan's equity market carries macro risks including currency depreciation, interest rate changes, and political uncertainty
• **Sector risk:** The {sector} sector has specific sensitivities to commodity prices, regulatory changes, and demand cycles
• **Company risk:** Individual company performance may be affected by management decisions, operational issues, or competitive pressures
• **Model risk:** This AI recommendation is probabilistic based on historical patterns — past signals do not guarantee future returns
• **Liquidity risk:** Some PSX stocks may have lower trading volumes, making entry/exit at desired prices difficult
• **Always invest only what you can afford to lose and consider consulting a financial advisor**

**SECTION 6 - BOTTOM LINE:**
• **Decision:** {rating['advice']}
• **Entry price:** PKR {price:,.2f} with target exit at PKR {price * 1.15:,.2f}
• **Stop-loss:** Hard stop at PKR {price * 0.88:,.2f} to protect capital
• **Risk/Reward ratio:** 1:1.25 (12% downside vs 15% upside potential)

{'=' * 60}
**⚠️ DISCLAIMER:** AI-generated analysis for educational purposes only. Not financial advice. Past performance does not guarantee future results.
"""


# ============================================
# LEGACY FUNCTIONS
# ============================================

def generate_investment_report(
    symbol: str,
    probability: float,
    risk_profile: str,
    allocation_percent: float,
    amount_pkr: float,
    latest_price: float,
    shares: int,
    shap_factors: list = None,
    target_return_percent: float = 15.0,
    target_profit_pkr: float = None,
    estimated_annual_risk_percent: float = None,
) -> str:
    llm = LLMReportGenerator()
    return llm.generate_simple_report(
        symbol=symbol,
        probability=probability,
        allocation_percent=allocation_percent,
        amount_pkr=amount_pkr,
        latest_price=latest_price,
        shares=shares,
        target_return_percent=target_return_percent,
        target_profit_pkr=target_profit_pkr,
        risk_level=risk_profile,
        factors=shap_factors,
    )


def generate_portfolio_summary(plan, risk_profile: str, investment_amount: float) -> str:
    return PortfolioSummaryGenerator.generate_summary(plan, risk_profile, investment_amount)


__all__ = [
    'LLMReportGenerator',
    'PlainEnglishTranslator',
    'PortfolioSummaryGenerator',
    'get_stock_insight',
    'get_company_info',
    'get_company_description_via_ai',
    'generate_investment_report',
    'generate_portfolio_summary',
    'set_mistral_api_key',
]