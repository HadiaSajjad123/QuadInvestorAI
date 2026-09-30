"""
Company Data Enricher - Automatically generates company details for all PSX stocks
"""

import pandas as pd
import json
from pathlib import Path
import re

class CompanyDataEnricher:
    """Automatically enriches company data for all stocks"""
    
    # Sector mapping based on symbol patterns
    SECTOR_PATTERNS = {
        'BANKING': ['HBL', 'MCB', 'UBL', 'NBP', 'ABL', 'BAFL', 'BAHL', 'MEBL', 'BOP', 'FABL', 'BIPL', 'JSBL', 'SILK'],
        'CEMENT': ['LUCK', 'DGKC', 'MLCF', 'CHCC', 'PIOC', 'ACPL', 'KOHC', 'FCCL', 'BWCL', 'DCL', 'FLYNG'],
        'FERTILIZER': ['FFC', 'FFBL', 'EFERT', 'ENGRO', 'FATIMA'],
        'OIL_GAS': ['OGDC', 'PPL', 'PSO', 'APL', 'ATRL', 'BYCO', 'MARI', 'SHEL', 'PRL', 'NRL', 'TPL', 'HASCOL', 'SFL'],
        'POWER': ['HUBC', 'KAPCO', 'TSPL', 'KOHP', 'FTSM', 'NCPL', 'GADT', 'SPWL', 'PKGP', 'EPQL', 'NEPL', 'PIBTL'],
        'TEXTILE': ['NML', 'NCL', 'GATM', 'ILP', 'TOWL', 'ASHT', 'SSOM', 'FASM', 'GATM', 'CWSM', 'DMTX', 'GADT', 'ICL', 'KOSM', 'SERT', 'SITC'],
        'AUTOMOBILE': ['INDU', 'PSMC', 'HCAR', 'SAZEW', 'DWTM', 'GHNI', 'ATBA', 'HINO', 'DFML', 'AGTL'],
        'CHEMICAL': ['SIBL', 'LOTCHEM', 'ICI', 'NRSL', 'DYNO', 'BIFO', 'BERG', 'SPL', 'SURC', 'AGRIC'],
        'TECHNOLOGY': ['SYS', 'AVN', 'NETSOL', 'AIRLINK', 'TRG', 'WTL', 'OCTOPUS', 'TELE', 'ZAHID'],
        'FOOD_BEVERAGE': ['NESTLE', 'UNITY', 'PMRS', 'SAPL', 'NATF', 'MFFL', 'SHNI', 'QUICE', 'HUMAN'],
        'PHARMA': ['GLAXO', 'ABOT', 'SEARLE', 'IBLHL', 'FEROZ', 'HINO', 'OVIS', 'AGP', 'BIO'],
        'STEEL': ['ASTL', 'ISL', 'ASML', 'MUGHAL', 'BAPL', 'KASL', 'ASL', 'AKBL', 'INIL'],
        'CABLE_ELECTRICAL': ['PACE', 'PAEL', 'SEL', 'SIEM', 'PICT'],
        'LEATHER': ['ITANZ', 'BDIL', 'SFL'],
        'ENGINEERING': ['ARCTM', 'BWHL', 'HWQS', 'MIRKS', 'DWSM', 'SBL', 'DEL', 'AMBL', 'STJT', 'CHAS', 'BELA'],
        'SUGAR': ['PMRS', 'SHAFI', 'ALNRS', 'JVDC', 'MSCL', 'SML', 'RMPL', 'TSMF', 'MSOT'],
        'INSURANCE': ['EFU', 'IGIHL', 'ICL', 'JLICL', 'PAKRI', 'AICL', 'AGICO', 'ATIL', 'CSIL', 'HICL'],
        'MODARABA': ['FECM', 'FEM', 'FEM', 'FFLM', 'FMC', 'FNEL', 'FRSM', 'FZCM', 'JSM', 'NCPM', 'OLPM', 'SINDM', 'TMM', 'TRSM', 'USLM'],
        'MUTUAL_FUNDS': ['ABLFS', 'AKIF', 'ALIF', 'ARTF', 'ASGF', 'ATIF', 'AWTIF', 'BCIF', 'CJIF', 'CPBF', 'CPSF', 'CYIF', 'DAIF', 'FBIF', 'FIBIF', 'FIF', 'FNBF', 'FPAF', 'FSVL', 'FWF', 'HIF', 'HWEF', 'HWF', 'ICCF', 'ICIF', 'IDIF', 'IICF', 'IMKF', 'ISETF', 'JSIF', 'JSMF', 'JSMIF', 'KSEETF', 'MCBFSF', 'MEF', 'MEPF', 'MHIF', 'MIIF', 'MZSHF', 'NAGIF', 'NBPIF', 'NBPILS', 'NBPISF', 'NBPNF', 'NBPREITF', 'NCDF', 'NICF', 'NIT', 'NITREITF', 'NLPF', 'PBAF', 'PICF', 'PIMEF', 'PISF', 'PMAF', 'PPAF', 'PPIF', 'PRSF', 'PSIF', 'RMIF', 'SIF', 'SIGF', 'SIPF', 'SPIF', 'SWIF', 'TBF', 'TGSF', 'TMF', 'UAMF', 'UAPF', 'UBSF', 'UFP', 'UIF', 'UILF', 'ULIF', 'UMF', 'UMSF', 'UNAF', 'UNBIAF', 'UNF', 'UNIMF', 'UNREITF', 'USF', 'UTIF', 'UWLF', 'WEIF', 'WFSF', 'WIF', 'WIP', 'WISF', 'WSSF', 'ZIEF', 'ZIGF', 'ZIPF', 'ZITF', 'ZJSF', 'ZLPF', 'ZMSF', 'ZVSF'],
        'REAL_ESTATE': ['REIT', 'DCR', 'GLA'],
        'MISCELLANEOUS': ['DAWH', 'PKGS', 'COLG', 'UNILEVER', 'PAKT', 'TREET', 'LOADS', 'PSEL', 'PNSC', 'SNGP', 'SSGC', 'SGML', 'GEMP', 'GAMON', 'MSCL', 'SPEL', 'THCCL', 'TPL', 'TRIPF', 'UNIC', 'URMT']
    }
    
    # Industry groups for better categorization
    INDUSTRY_GROUPS = {
        'BANKING': 'Financial Services',
        'INSURANCE': 'Financial Services',
        'MODARABA': 'Financial Services',
        'MUTUAL_FUNDS': 'Asset Management',
        'CEMENT': 'Construction Materials',
        'STEEL': 'Metals & Mining',
        'FERTILIZER': 'Agriculture',
        'OIL_GAS': 'Energy',
        'POWER': 'Utilities',
        'TEXTILE': 'Textiles & Apparel',
        'AUTOMOBILE': 'Automotive',
        'CHEMICAL': 'Chemicals',
        'TECHNOLOGY': 'Technology',
        'FOOD_BEVERAGE': 'Consumer Goods',
        'PHARMA': 'Healthcare',
        'CABLE_ELECTRICAL': 'Electrical Equipment',
        'LEATHER': 'Leather Goods',
        'ENGINEERING': 'Industrial Goods',
        'SUGAR': 'Agriculture',
        'REAL_ESTATE': 'Real Estate',
    }
    
    @classmethod
    def detect_sector(cls, symbol: str) -> str:
        """Detect sector based on symbol pattern"""
        symbol_upper = symbol.upper()
        
        for sector, symbols in cls.SECTOR_PATTERNS.items():
            if symbol_upper in symbols or any(s in symbol_upper for s in symbols[:3]):
                return sector.replace('_', ' ').title()
        
        # Check for mutual fund pattern (ends with F, FS, IF, etc.)
        if symbol_upper.endswith('F') or symbol_upper.endswith('FS') or 'FUND' in symbol_upper:
            return 'Mutual Funds'
        
        # Check for modaraba pattern
        if symbol_upper.endswith('M') or 'MOD' in symbol_upper:
            return 'Modaraba'
        
        return 'PSX Listed'
    
    @classmethod
    def get_industry_group(cls, sector: str) -> str:
        """Get broader industry group"""
        return cls.INDUSTRY_GROUPS.get(sector.upper().replace(' ', '_'), 'Diversified')
    
    @classmethod
    def generate_company_name(cls, symbol: str, sector: str) -> str:
        """Generate a sensible company name"""
        symbol_upper = symbol.upper()
        
        # Known company name patterns
        if symbol_upper == 'AABS':
            return 'Aabs' + ' ' + sector
        elif symbol_upper == 'AGSML':
            return 'Agriauto' + ' ' + sector
        elif symbol_upper == 'AKDHL':
            return 'AKD' + ' ' + sector
        # Add more as needed
        
        # Default: symbol + sector
        if sector != 'PSX Listed':
            return f"{symbol_upper} ({sector})"
        return symbol_upper
    
    @classmethod
    def generate_description(cls, symbol: str, sector: str) -> str:
        """Generate a meaningful description based on sector"""
        symbol_upper = symbol.upper()
        
        descriptions = {
            'Banking': f"{symbol_upper} is a commercial bank operating in Pakistan, offering retail, corporate, and digital banking services to customers nationwide.",
            'Cement': f"{symbol_upper} is a cement manufacturer in Pakistan, producing high-quality cement for the construction industry and infrastructure projects.",
            'Fertilizer': f"{symbol_upper} is a fertilizer company serving Pakistan's agricultural sector with essential plant nutrients and crop solutions.",
            'Oil & Gas': f"{symbol_upper} is engaged in Pakistan's energy sector, involved in oil and gas exploration, production, refining, or marketing.",
            'Power': f"{symbol_upper} is an independent power producer (IPP) generating electricity for Pakistan's national grid.",
            'Textile': f"{symbol_upper} is a textile manufacturer producing yarn, fabric, and garments for domestic and international markets.",
            'Automobile': f"{symbol_upper} is involved in automobile assembly, manufacturing, or parts production in Pakistan.",
            'Chemical': f"{symbol_upper} manufactures industrial chemicals for various applications in Pakistan's industrial sector.",
            'Technology': f"{symbol_upper} provides IT services, software solutions, and technology products for businesses in Pakistan.",
            'Pharma': f"{symbol_upper} is a pharmaceutical company manufacturing medicines and healthcare products for the Pakistani market.",
            'Steel & Metals': f"{symbol_upper} is a steel manufacturer producing products for Pakistan's construction and industrial sectors.",
            'Cables & Electrical': f"{symbol_upper} manufactures electrical cables, wires, and power transmission equipment.",
            'Leather': f"{symbol_upper} is a leather manufacturer producing finished leather for footwear, garments, and upholstery.",
            'Engineering': f"{symbol_upper} manufactures industrial equipment, machinery, and engineering products.",
            'Sugar': f"{symbol_upper} is a sugar manufacturer and distillery producing refined sugar and industrial alcohol.",
            'Insurance': f"{symbol_upper} provides insurance and risk management services to individuals and businesses in Pakistan.",
            'Modaraba': f"{symbol_upper} is a modaraba company engaged in Islamic financial services and investments.",
            'Mutual Funds': f"{symbol_upper} is a mutual fund that pools investor money to invest in diversified securities.",
            'Utilities': f"{symbol_upper} is a utility company providing essential services like gas distribution.",
            'Hospitality': f"{symbol_upper} operates hotels, resorts, and hospitality services in Pakistan.",
            'Shipping': f"{symbol_upper} is a shipping company operating cargo and transport services.",
            'Manufacturing': f"{symbol_upper} is a diversified manufacturer of consumer and industrial products.",
            'Footwear': f"{symbol_upper} manufactures footwear products for domestic and export markets.",
            'Electronics': f"{symbol_upper} manufactures electronic appliances and consumer electronics.",
            'Packaging': f"{symbol_upper} is a packaging materials manufacturer producing paper and board products.",
            'FMCG': f"{symbol_upper} manufactures consumer goods including personal care and food products.",
            'Tobacco': f"{symbol_upper} is involved in tobacco processing and cigarette manufacturing.",
            'Conglomerate': f"{symbol_upper} is a diversified holding company with investments across multiple sectors.",
            'Automotive Parts': f"{symbol_upper} manufactures automotive components and parts for the auto industry.",
        }
        
        base_desc = descriptions.get(sector, f"{symbol_upper} is listed on the Pakistan Stock Exchange (PSX).")
        
        # Add specific details for known stocks
        specific_descs = {
            'DFML': "Dewan Farooque Motors is an automobile manufacturer producing vehicles in Pakistan.",
            'WTL': "WorldCall Telecom provides telecommunication and broadband services across Pakistan.",
            'TRG': "TRG Pakistan is a business process outsourcing (BPO) company with global operations.",
            'TELE': "Telecard Limited is a telecommunications company offering data and voice services.",
            'SNGP': "Sui Northern Gas Pipelines distributes natural gas to northern Pakistan.",
            'SSGC': "Sui Southern Gas Company distributes natural gas to Sindh and Balochistan.",
            'PSEL': "Pakistan Services operates the Pearl Continental hotel chain across Pakistan.",
            'PNSC': "Pakistan National Shipping Corporation manages Pakistan's national shipping fleet.",
            'PKGS': "Packages Limited is Pakistan's largest packaging materials manufacturer.",
            'TREET': "Treet Corporation manufactures razors, blades, batteries, and other consumer products.",
            'PAEL': "Pakistan Elektron manufactures PEL appliances including ACs and refrigerators.",
            'LOADS': "Service Industries is Pakistan's largest footwear manufacturer and exporter.",
            'PAKT': "Pakistan Tobacco is the Pakistani affiliate of British American Tobacco.",
        }
        
        if symbol_upper in specific_descs:
            return specific_descs[symbol_upper]
        
        return base_desc


def update_company_directory(live_df: pd.DataFrame = None):
    """
    Update the PSX_COMPANY_DIRECTORY with all missing stocks
    """
    # Load existing directory
    existing_dir = {}
    # This will be updated with your existing directory
    
    # Get all symbols from live data
    if live_df is None:
        return
    
    all_symbols = live_df['symbol'].unique()
    missing_symbols = [s for s in all_symbols if s not in existing_dir]
    
    print(f"Total symbols: {len(all_symbols)}")
    print(f"Existing entries: {len(existing_dir)}")
    print(f"Missing entries: {len(missing_symbols)}")
    
    # Generate missing entries
    new_entries = {}
    for symbol in missing_symbols:
        sector = CompanyDataEnricher.detect_sector(symbol)
        new_entries[symbol] = {
            "name": CompanyDataEnricher.generate_company_name(symbol, sector),
            "sector": sector,
            "desc": CompanyDataEnricher.generate_description(symbol, sector),
            "industry_group": CompanyDataEnricher.get_industry_group(sector)
        }
    
    return new_entries


def update_report_generator_with_missing_companies():
    """
    Generate Python code to add missing companies to report_generator.py
    """
    # This will generate the missing entries in the correct format
    pass


if __name__ == "__main__":
    # Load live data
    from src.config import LIVE_MARKET_CSV
    if LIVE_MARKET_CSV.exists():
        live_df = pd.read_csv(LIVE_MARKET_CSV)
        all_symbols = live_df['symbol'].unique()
        
        print("\n" + "=" * 70)
        print("PSX COMPANY DATA ENRICHER")
        print("=" * 70)
        
        # Categorize all symbols
        categorized = {}
        for symbol in all_symbols:
            sector = CompanyDataEnricher.detect_sector(symbol)
            if sector not in categorized:
                categorized[sector] = []
            categorized[sector].append(symbol)
        
        print("\n📊 Sector Distribution:")
        print("-" * 50)
        for sector, symbols in sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {sector:<25}: {len(symbols):>3} stocks")
        
        print("\n" + "=" * 70)
        print("✅ Ready to generate missing company data")
        print("=" * 70)