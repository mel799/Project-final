from src.data_loader.load_temp import load_temperature
from src.data_loader.load_pop import load_population
from src.data_loader.load_gdp import load_gdp
from src.data_loader.load_elec_consumption import load_consumption
from src.data_loader.pop_yearly_monthly import make_population_monthly
from src.data_loader.GDP_monthly import make_gdp_monthly

from src.data_preprocessing.temp_heat_need import add_heat_need
from src.data_preprocessing.add_features import add_features
from src.data_preprocessing.merge_data import merge_all
# file name in import should match the actual file name 
# second term is the name of the function inside that file

def main():

    print("=== Loading temperature data ===")
    temp_df = load_temperature()
    print("=== Adding heat need ===")
    temp_df = add_heat_need(temp_df)

    print("=== Loading population data ===")
    pop_df = load_population()
    print("=== Converting population to monthly ===")
    pop_monthly = make_population_monthly()

    print("=== Loading GDP data ===")
    gdp_df = load_gdp()
    print("=== Converting GDP to monthly ===")
    gdp_monthly = make_gdp_monthly()

    print("=== Loading electricity consumption data ===")
    cons_df = load_consumption()

    print("=== Merging all datasets ===")
    merged_df = merge_all(temp_df, pop_monthly, gdp_monthly, cons_df)

    print("=== Adding features ===")
    final_df = add_features(merged_df)
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
