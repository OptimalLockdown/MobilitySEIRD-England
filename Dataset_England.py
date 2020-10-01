import numpy as np
import pandas as pd
from datetime import datetime
import io
import requests
from uk_covid19 import Cov19API

## This works
# country = 'Austria'
# country = 'France'
# country = 'Germany'
# country = 'Spain'
country = 'England'
# country = 'Scotland'
# country = 'Northern Ireland'

## Following still not considered
# country = 'Scotland'
# country = 'Belgium'
# country = 'Denmark'
# country = 'Netherlands'
# country = 'Norway'
# country = 'Portugal'
# country = 'South Korea'
# country = 'Sweden'
# country = 'Ukraine'
# country = 'United States'

start_date = "1Mar"
end_date = "31Aug"

subfolder = f"{country.lower()}_inference_data_{start_date}_to_{end_date}"
# ############################ Reading contact matrix data #######################
# These are contact matrices for 5 years age groups, up to the age of 75, and a single age group from 75+. By comparing
# with the plots in the original Prem's work, I think each column represents the individual, and each row the number of
# contacts that the individual has with each age group.
# If you convert the dataframe to numpy array, the first index refers to the individual, and the second to the contacts
# with other age groups. This is what we want in our formulation.
# Note: entries are age-stratified expected number of contacts per day.
if country in ['Austria', 'France', 'Germany', 'Italy']:
    fileno, headertype = '1', 0
else:
    fileno, headertype = '2', None

if country in ['England', 'Scotland', 'Northern Ireland']:
    country_contact = 'United Kingdom of Great Britain'
else:
    country_contact = country
contact_matrix_all_locations = pd.read_excel(
    "data/contact_matrices_152_countries/MUestimates_all_locations_" + fileno + ".xlsx"
    , sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_home = pd.read_excel("data/contact_matrices_152_countries/MUestimates_home_" + fileno + ".xlsx",
                                    sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_school = pd.read_excel("data/contact_matrices_152_countries/MUestimates_school_" + fileno + ".xlsx",
                                      sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_work = pd.read_excel("data/contact_matrices_152_countries/MUestimates_work_" + fileno + ".xlsx",
                                    sheet_name=country_contact, header=headertype).to_numpy()
contact_matrix_other_locations = pd.read_excel(
    "data/contact_matrices_152_countries/MUestimates_other_locations_" + fileno + ".xlsx",
    sheet_name=country_contact, header=headertype).to_numpy()
########################################################################################################################
################################ United Nation World Population Data by age #############################################
if country in ['England', 'Scotland', 'Northern Ireland']:
    country_UN_data = 'United Kingdom of Great Britain and Northern Ireland'
else:
    country_UN_data = country

UN_Pop_Data = pd.read_csv("data/UNdata_Export_20200820_181107223.csv")
## Choose both the sexes
UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Sex"] == 'Both Sexes']
## Choose the total area
UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Area"] == 'Total']
# Choose the country
UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Country or Area"] == country_UN_data]
# Choose the most recent year
if UN_Pop_Data[UN_Pop_Data["Year"] == '2018'].empty:
    UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Year"] == 2018]
else:
    UN_Pop_Data = UN_Pop_Data[UN_Pop_Data["Year"] == '2018']
Country_age_data_list = []
age_list = ['0 - 4', '5 - 9', '10 - 14', '15 - 19', '20 - 24', '25 - 29', '30 - 34', '35 - 39', '40 - 44', '45 - 49',
            '50 - 54', '55 - 59', '60 - 64', '65 - 69', '70 - 74', 'Total']
age_list_lb = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
for x in age_list:
    if (UN_Pop_Data['Age'] == x).any():
        Country_age_data_list.append(UN_Pop_Data[UN_Pop_Data['Age'] == x]['Value'].iloc[0])
Country_age_data_list[-1] = Country_age_data_list[-1] - sum(Country_age_data_list[:15])
Country_age_weight = Country_age_data_list / sum(Country_age_data_list)

########################################################################################################################
################################ Countrywise Data for age-structured death #############################################
if country == 'France':
    sheetname, rows = 'SpF_by age and sex_HospitalData', 9
elif country == 'England & Wales':
    ## Weekly death data which accounts for deaths in hospital, hospice and other places
    sheetname, rows = 'ONS_WeeklyOccurrenceDeaths', 19
elif country == 'England':
    sheetname, rows = 'NHS_Daily_Data', 7
elif country == 'Scotland':
    sheetname, rows = 'NRS_Age_&_Sex', 7
elif country == 'Spain':
    sheetname, rows = 'MSCBS_Data', 9
elif country == 'Germany':
    sheetname, rows = 'Daily Report RKI_Data', 10
elif country == 'Austria':
    sheetname, rows = 'EMS_Data', 10
elif country == 'Belgium':
    sheetname, rows = 'Deaths_by_occurrence', '6'

if country == 'England':
    file = "data/Deaths-Age-Sex_Covid-19_EnglandWales_01-09.xlsx"
    data = pd.read_excel(file, sheet_name=sheetname, skiprows=[0, 1, 2, 3, 4], nrows=rows, header=None)
    days = data.iloc[0, 4:]
    dates_yday = [(datetime.strptime(str(x)[:10].replace('.', '-'), '%Y-%m-%d').timetuple().tm_yday) for x in days]
    death_data = data.iloc[2:, 4:].to_numpy()
    age_groups = data.iloc[2:, 0].to_numpy()
    age_groups_lb = [int(x.replace('+', '-').split("-")[0]) for x in age_groups]
    total_population_age_group = data.iloc[2:, 1].to_numpy()
else:
    file = "data/covid_pooled_25_08.csv"
    data = pd.read_csv(file)

    data = data[data["country"] == country][data["excelsheet"] == sheetname]
    days = data["death_reference_date"].unique()
    dates_yday = [int(datetime.strptime(str(x).replace('.', '-'), '%d-%m-%Y').timetuple().tm_yday) for x in days]
    age_groups = data[data["death_reference_date"] == days[0]].iloc[:rows]["age_group"]
    age_groups_lb = [int(age_groups.iloc[x].replace('+', '-').split("-")[0]) for x in range(len(age_groups))]
    death_data, collected = [], []
    for ind in range(len(days)):
        if len(data[data["death_reference_date"] == days[ind]].iloc[:rows]["cum_death_both"].to_numpy(
                dtype=float)) == len(age_groups):
            death_data.append(
                data[data["death_reference_date"] == days[ind]].iloc[:rows]["cum_death_both"].to_numpy(dtype=float))
            collected.append(1)
        else:
            collected.append(0)
    death_data = np.array(death_data).transpose()[np.argsort(age_groups_lb)]
    age_groups = age_groups.to_numpy()[np.argsort(age_groups_lb)]
    age_groups_lb = np.array(age_groups_lb)[np.argsort(age_groups_lb)]
    total_population_age_group = data[data["death_reference_date"] == days[0]].iloc[:rows]["pop_both"].to_numpy()[
        np.argsort(age_groups_lb)]
    dates_yday = [dates_yday[x] for x in np.nonzero(collected)[0]]

# print(age_groups, age_groups_lb, death_data)

# FOR ENGLAND: NEED TO USE OLD CONTACT MATRICES CREATED WITH THE NOTEBOOK-

#
# ## Modify contact matrix using the age_groups_lb ###
# Indices = []
# death_data_used = []
# age_used_lb = []
# total_population_age_group_used = []
# for x in range(len(age_groups_lb)-1):
#     if age_groups_lb[x+1] in age_list_lb:
#         Indices.append(list(range(age_list_lb.index(age_groups_lb[x]),age_list_lb.index(age_groups_lb[x+1]))))
#         death_data_used.append(death_data[x,:])
#         age_used_lb.append(age_groups_lb[x])
#         total_population_age_group_used.append(total_population_age_group[x])
#     else:
#         if age_groups_lb[x] in age_list_lb:
#             Indices.append(list(range(age_list_lb.index(age_groups_lb[x]), len(age_list_lb))))
#             death_data_used.append(np.sum(death_data[x:,:],axis=0))
#             total_population_age_group_used.append(np.sum(total_population_age_group[x:],axis=0))
#             age_used_lb.append(age_groups_lb[x])
#
# print(age_used_lb, total_population_age_group_used, death_data_used)
#
# ## We reweight/compute the contact matrices based on the age-structure of the death data
# def reweight_contact_matrix(contact_matrix):
#     contact_matrix_new = np.zeros(shape=contact_matrix.shape)
#     for ind_columns in range(len(Indices)):
#         for ind_rows in range(len(Indices)):
#             contact_matrix_new[ind_rows,ind_columns] = np.average(np.sum(contact_matrix[:,Indices[ind_columns]],axis=1)[Indices[ind_rows]],
#                                                             weights=Country_age_weight[Indices[ind_rows]])
#     return contact_matrix_new
#
# contact_matrix_home = reweight_contact_matrix(contact_matrix_home)
# contact_matrix_school = reweight_contact_matrix(contact_matrix_school)
# contact_matrix_work = reweight_contact_matrix(contact_matrix_work)
# contact_matrix_all_locations = reweight_contact_matrix(contact_matrix_all_locations)
# contact_matrix_other_locations = reweight_contact_matrix(contact_matrix_other_locations)
#
# np.save('data/' + subfolder + '/contact_matrix_home_{}.npy'.format(country.lower()), contact_matrix_home)
# np.save('data/' + subfolder + '/contact_matrix_work_{}.npy'.format(country.lower()), contact_matrix_work)
# np.save('data/' + subfolder + '/contact_matrix_school_{}.npy'.format(country.lower()), contact_matrix_school)
# np.save('data/' + subfolder + '/contact_matrix_other_{}.npy'.format(country.lower()), contact_matrix_other_locations)

# np.save('data/' + subfolder + '/tot_pop_age_group.npy', total_population_age_group_used)
np.save('data/' + subfolder + f'/{country.lower()}_pop.npy', total_population_age_group)

####################### Final Output : Death_Data ############################################
##### de-cumsum of death data ####
death_data = np.array(death_data)
death_data_decum = np.zeros(shape=death_data.shape)
death_data_decum[:, -1] = death_data[:, -1]
for ind in range(0, death_data.shape[1] - 1):
    death_data_decum[:, ind] = death_data[:, ind] - death_data[:, ind + 1]

Death_Data = np.concatenate((np.array(dates_yday).reshape(1, -1), np.array(death_data_decum)))

# ########################################################################################################################
# ################################ Reading Countrywise Data for confirmed infected/Admitted/Hospitalized/Ventilator people ########################################
if country in ['England', 'Scotland', 'Northern Ireland']:
    # def get_data(url):
    #     response = requests.get(endpoint, timeout=10)
    #     if response.status_code >= 400:
    #         raise RuntimeError(f'Request failed: {response.text}')
    #     return response.json()

    if country == 'Scotland':
        Scotland_only = ['areaType=nation', 'areaName=Scotland']
        cases_and_deaths = {"date": "date",
                            "Infected": "newCasesByPublishDate",
                            "Admission": "newAdmissions",
                            "Hospitalized": "hospitalCases",
                            "OccupiedVB": "covidOccupiedMVBeds"
                            }
        api = Cov19API(filters=Scotland_only, structure=cases_and_deaths)
        data = api.get_json()
        # endpoint = (
        #         'https://api.coronavirus.data.gov.uk/v1/data?'
        #         'filters=areaType=nation;areaName=scotland&'
        #         'structure={"date":"date","Infected":"cumCasesByPublishDate", "Admission":"cumAdmissions", "Hospitalized":"hospitalCases", "OccupiedVB":"covidOccupiedMVBeds"}')
        # data = get_data(endpoint)
        date, Infected, Admission, Hospitalized, OccupiedVB = [], [], [], [], []
        for ind in range(data['length']):
            date.append(int(datetime.strptime(str(data['data'][ind]['date']), '%Y-%m-%d').timetuple().tm_yday))
            Infected.append(data['data'][ind]['Infected'])
            Admission.append(data['data'][ind]['Admission'])
            Hospitalized.append(data['data'][ind]['Hospitalized'])
            OccupiedVB.append(data['data'][ind]['OccupiedVB'])
    elif country == 'Northern Ireland':
        NI_only = ['areaType=nation', 'areaName=Northern Ireland']
        cases_and_deaths = {"date": "date",
                            "Infected": "newCasesByPublishDate",
                            "Admission": "newAdmissions",
                            "Hospitalized": "hospitalCases",
                            "OccupiedVB": "covidOccupiedMVBeds"
                            }
        api = Cov19API(filters=NI_only, structure=cases_and_deaths)
        data = api.get_json()
        # endpoint = (
        #         'https://api.coronavirus.data.gov.uk/v1/data?'
        #         'filters=areaType=nation;areaName=northern ireland&'
        #         'structure={"date":"date","Infected":"cumCasesByPublishDate", "Admission":"cumAdmissions", "Hospitalized":"hospitalCases", "OccupiedVB":"covidOccupiedMVBeds"}')
        # data = get_data(endpoint)
        date, Infected, Admission, Hospitalized, OccupiedVB = [], [], [], [], []
        for ind in range(data['length']):
            date.append(int(datetime.strptime(str(data['data'][ind]['date']), '%Y-%m-%d').timetuple().tm_yday))
            Infected.append(data['data'][ind]['Infected'])
            Admission.append(data['data'][ind]['Admission'])
            Hospitalized.append(data['data'][ind]['Hospitalized'])
            OccupiedVB.append(data['data'][ind]['OccupiedVB'])
    elif country == 'England':
        england_only = ['areaType=nation', 'areaName=England']
        cases_and_deaths = {"date": "date",
                            "Infected": "newCasesByPublishDate",
                            "Admission": "newAdmissions",
                            "Hospitalized": "hospitalCases",
                            "OccupiedVB": "covidOccupiedMVBeds"
                            }
        api = Cov19API(filters=england_only, structure=cases_and_deaths)
        data = api.get_json()
        # endpoint = (
        #     'https://api.coronavirus.data.gov.uk/v1/data?'
        #     'filters=areaType=nation;areaName=england&'
        #     'structure={"date":"date","Infected":"cumCasesByPublishDate", "Admission":"cumAdmissions", "Hospitalized":"hospitalCases", "OccupiedVB":"covidOccupiedMVBeds"}')
        # data = get_data(endpoint)
        date, Infected, Admission, Hospitalized, OccupiedVB = [], [], [], [], []
        for ind in range(data['length']):
            date.append(int(datetime.strptime(str(data['data'][ind]['date']), '%Y-%m-%d').timetuple().tm_yday))
            Infected.append(data['data'][ind]['Infected'])
            Admission.append(data['data'][ind]['Admission'])
            Hospitalized.append(data['data'][ind]['Hospitalized'])
            OccupiedVB.append(data['data'][ind]['OccupiedVB'])
    # Other_Data = np.vstack((date, Infected, Admission, Hospitalized, OccupiedVB))
    Other_Data = np.vstack((date, Hospitalized))
else:
    url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"
    s = requests.get(url).content
    JH_Data = pd.read_csv(io.StringIO(s.decode('utf-8')))
    JH_Date_country = JH_Data[JH_Data["Country"] == country]

    dates_yday = [int(datetime.strptime(str(x)[:10], '%Y-%m-%d').timetuple().tm_yday) for x in JH_Date_country["Date"]]
    confirmed = JH_Date_country["Confirmed"].to_numpy(dtype=int)
    deaths = JH_Date_country["Deaths"].to_numpy(dtype=int)
    recovered = JH_Date_country["Recovered"].to_numpy(dtype=int)
    Other_Data = np.vstack((dates_yday, confirmed, recovered))


##################################### Final Output: Other_data #########################################################

########################################################################################################################
################################## Merge Death and Confirmed/Admitted data ################################################################
def merging_function(death, confirmed):
    start_day, end_day = min(np.min(death[0, :]), np.min(confirmed[0, :])), max(np.max(death[0, :]),
                                                                                np.max(confirmed[0, :]))
    Data = [[ind] for ind in range(int(start_day), int(end_day) + 1)]
    for ind in range(len(Data)):
        if Data[ind][0] in death[0, :]:
            Data[ind] = Data[ind] + death[1:, np.where(death[0, :] == Data[ind][0])[0]].reshape(-1, ).tolist()
        else:
            for i in range(death.shape[0] - 1):
                Data[ind].append('None')
        if Data[ind][0] in confirmed[0, :]:
            Data[ind] = Data[ind] + confirmed[1:, np.where(confirmed[0, :] == Data[ind][0])[0]].reshape(-1, ).tolist()
        else:
            for i in range(confirmed.shape[0] - 1):
                Data[ind].append('None')

    return np.transpose(Data)


if country == 'England':
    Observed_Data = merging_function(Death_Data, Other_Data)
else:
    Observed_Data = merging_function(Death_Data, Other_Data)

## Keep Observed Data only from 1st March
dynamics_start_day = int(datetime.strptime(str('2020-03-01')[:10], '%Y-%m-%d').timetuple().tm_yday)
index_dynamics_start_day = Observed_Data[0, :].tolist().index(dynamics_start_day)
Observed_Data = np.delete(Observed_Data, np.arange(0, index_dynamics_start_day), axis=1)

## Keep Observed Data until 31st August
dynamics_end_day = int(datetime.strptime(str('2020-08-31')[:10], '%Y-%m-%d').timetuple().tm_yday)
index_dynamics_end_day = Observed_Data[0, :].tolist().index(dynamics_end_day) + 1
last_day = Observed_Data[0, :].tolist().index(Observed_Data[0, -1]) + 1
Observed_Data = np.delete(Observed_Data, np.arange(index_dynamics_end_day, last_day), axis=1).transpose()[:, 1:]

np.save('data/' + subfolder + '/observed_data.npy', Observed_Data)

###################################### Final Output: Observed_data ######################################################

########################################################################################################################
################################## Reading Google mobility data ################################################################
if country in ['England', 'Scotland', 'Northern Ireland', 'England & Wales']:
    country_mobility = 'United Kingdom'
    start_date_lockdown = 77  ## 17th March in yday format
else:
    country_mobility = country
    ## Load lockadown dates for each countroes
    lockdown_dates = pd.read_csv("data/Lockdown_Dates.csv")
    lockdown_dates = lockdown_dates[lockdown_dates["Level "] == 'National ']
    start_date_lockdown = int(datetime.strptime(
        lockdown_dates[lockdown_dates["Countries and territories "] == ' ' + country + ' ']["Start date "].iloc[0][:10],
        '%Y-%m-%d').timetuple().tm_yday)
    end_date_lockdown = int(datetime.strptime(
        lockdown_dates[lockdown_dates["Countries and territories "] == ' ' + country + ' ']["End date "].iloc[0][:10],
        '%Y-%m-%d').timetuple().tm_yday)

global_mobility = pd.read_csv("data/Global_Mobility_Report.csv")
global_mobility = global_mobility[global_mobility["country_region"] == country_mobility]
global_mobility_whole = global_mobility[global_mobility["sub_region_1"].isna()]
global_mobility_whole = global_mobility_whole.set_index(global_mobility_whole["date"])
date = [int(datetime.strptime(str(x)[:10], '%Y-%m-%d').timetuple().tm_yday) for x in global_mobility_whole["date"]]

## Keep mobility only from 1st March to 31st August
dynamics_start_day = int(datetime.strptime(str('2020-03-01')[:10], '%Y-%m-%d').timetuple().tm_yday)
index_dynamics_start_day = date.index(dynamics_start_day)
dynamics_end_day = int(datetime.strptime(str('2020-09-01')[:10], '%Y-%m-%d').timetuple().tm_yday)
index_dynamics_end_day = date.index(dynamics_end_day)
final_day = date.index(date[-1]) + 1

dates_to_remove = np.hstack((np.arange(0, index_dynamics_start_day), np.arange(index_dynamics_end_day, final_day)))

## indices of the lockdown day
index_start_lockdown_day = date.index(start_date_lockdown)

mobility_data_residential_raw = global_mobility_whole["residential_percent_change_from_baseline"].to_frame().transpose()
mobility_data_workplaces_raw = global_mobility_whole["workplaces_percent_change_from_baseline"].to_frame().transpose()

# data constituting the "other" category
mobility_data_parks_raw = global_mobility_whole["parks_percent_change_from_baseline"].to_frame().transpose()
mobility_data_retail_and_recreation_raw = global_mobility_whole[
    "retail_and_recreation_percent_change_from_baseline"].to_frame().transpose()
mobility_data_transit_stations_raw = global_mobility_whole[
    "transit_stations_percent_change_from_baseline"].to_frame().transpose()
mobility_data_grocery_and_pharmacy_raw = global_mobility_whole[
    "grocery_and_pharmacy_percent_change_from_baseline"].to_frame().transpose()

from scipy.signal import savgol_filter


def transform_alpha_df(df, window):
    zeros = pd.DataFrame(data=np.zeros((1, index_start_lockdown_day)), columns=df.columns[:index_start_lockdown_day])
    data = savgol_filter(df.to_numpy().reshape(-1), window, 1)
    y = pd.DataFrame(data.reshape(1, -1), columns=df.columns).iloc[:, index_start_lockdown_day:]
    return pd.concat((zeros, y), axis=1)


mobility_data_residential = transform_alpha_df(mobility_data_residential_raw, 15)
mobility_data_workplaces = transform_alpha_df(mobility_data_workplaces_raw, 13)
mobility_data_parks = transform_alpha_df(mobility_data_parks_raw, 11)
mobility_data_retail_and_recreation = transform_alpha_df(mobility_data_retail_and_recreation_raw, 11)
mobility_data_transit_stations = transform_alpha_df(mobility_data_transit_stations_raw, 11)
mobility_data_grocery_and_pharmacy = transform_alpha_df(mobility_data_grocery_and_pharmacy_raw, 11)

# Now transform that into the alpha multipliers; in order to form alpha_other, we combine data from the last 4 categories
# above, and assume contacts in park matter for 10%, while the others matter for 30% each.

mobility_home_raw = 1 + mobility_data_residential_raw / 100.0
mobility_work_raw = 1 + mobility_data_workplaces_raw / 100.0
mobility_parks_raw = 1 + mobility_data_parks_raw / 100.0
mobility_retail_and_recreation_raw = 1 + mobility_data_retail_and_recreation_raw / 100.0
mobility_transit_stations_raw = 1 + mobility_data_transit_stations_raw / 100.0
mobility_grocery_and_pharmacy_raw = 1 + mobility_data_grocery_and_pharmacy_raw / 100.0

alpha_home = 1 + mobility_data_residential / 100.0
alpha_work = 1 + mobility_data_workplaces / 100.0
alpha_parks = 1 + mobility_data_parks / 100.0
alpha_retail_and_recreation = 1 + mobility_data_retail_and_recreation / 100.0
alpha_transit_stations = 1 + mobility_data_transit_stations / 100.0
alpha_grocery_and_pharmacy = 1 + mobility_data_grocery_and_pharmacy / 100.0
alpha_other = 0.1 * alpha_parks + 0.3 * alpha_retail_and_recreation + 0.3 * alpha_transit_stations + 0.3 * alpha_grocery_and_pharmacy

# For the schools: we reduce to alpha=0.1 for days from the 22nd March.
data = np.ones((1, alpha_other.shape[1]))
data[:, index_start_lockdown_day + 5:] = 0.1
alpha_school = pd.DataFrame(data=data, columns=alpha_other.columns)


# Transform to numpy and add final steady states (more days with same value as the last day, up to the day for which we
# have observed data

def df_alphas_to_np(df, extra_number_days):
    array = np.zeros(df.shape[1] + extra_number_days)
    array[:df.shape[1]] = df
    array[df.shape[1]:] = df.iloc[0, -1]
    return array


alpha_home_np = df_alphas_to_np(alpha_home, 0)
alpha_work_np = df_alphas_to_np(alpha_work, 0)
alpha_other_np = df_alphas_to_np(alpha_other, 0)
alpha_school_np = df_alphas_to_np(alpha_school, 0)

# Dynamics starts on 1st March (Rmove data before 1st March) and until 31st Aug
alpha_home_np = np.delete(alpha_home_np, dates_to_remove)
alpha_work_np = np.delete(alpha_work_np, dates_to_remove)
alpha_other_np = np.delete(alpha_other_np, dates_to_remove)
alpha_school_np = np.delete(alpha_school_np, dates_to_remove)
# Dynamics starts on 1st March (Remove data before 1st March)

# np.save('data/' + subfolder + '/mobility_date', list(range(min(date), max(Observed_Data[0, :]) + 1)))
np.save('data/' + subfolder + '/mobility_home', alpha_home_np)
np.save('data/' + subfolder + '/mobility_work', alpha_work_np)
np.save('data/' + subfolder + '/mobility_other', alpha_other_np)
np.save('data/' + subfolder + '/mobility_school', alpha_school_np)
