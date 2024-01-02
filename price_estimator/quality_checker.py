'''
Created on Nov 29, 2023

@author: albertogallini
'''

from  const_and_utils import *
from datetime import datetime, timedelta
import logging

class QualityChecker(object):
    '''
    classdocs
    '''

    def __init__(self, report_folder: str, prevision_date: datetime, logger: logging.Logger ):
        
        self.report_folder = report_folder
        self.prevision_date_str = prevision_date.strftime("%d-%m-%Y")
        if prevision_date.weekday() == 4:
            next_date = prevision_date + timedelta(days=3)
        else:
            next_date = prevision_date + timedelta(days=1)
            
        self.next_date_str = next_date.strftime("%d-%m-%Y")
        self.prevision_folder = report_folder + self.prevision_date_str + "/"
        self.next_date_floder = report_folder + self.next_date_str + "/"
        self.logger = logger
        
        
    
    def __load_data(self):
        import os
        import pandas as pd
        from scipy.stats import norm
        self.logger.info("loading data for " + self.prevision_folder +" -> " +self.next_date_floder)
        self.previous_day_price = dict()
        self.prices_to_match = dict()
        self.estimates       = dict()
        
        files = os.listdir(self.next_date_floder)
        files = [file for file in files if 'est_prices_' in file]
        for f in files:
            try:
                price_to_match = pd.read_csv(self.next_date_floder+f)['prices'].iloc[-1]
                self.prices_to_match[get_ticker(f)[0]] = price_to_match
            except Exception as e:
                print("Caught an exception: ", e)
                print("Error loading data")
                continue
            
            
        files = os.listdir(self.prevision_folder)
        files = [file for file in files if 'est_prices_' in file]
        for f in files:
            try:
                previous_day_price = pd.read_csv(self.next_date_floder+f)['prices'].iloc[-1]
                self.previous_day_price[get_ticker(f)[0]] = previous_day_price
                fprices = list(pd.read_csv(self.prevision_folder+f)["prices"].values)
                fprices =fprices[:-1]
                mu, std = norm.fit(fprices)
                self.estimates[get_ticker(f)[0]] = (mu,std)
        
            except Exception as e:
                print("Caught an exception: ", e)
                print("Error loading data")
                continue
         
        self.logger.info("actuals:") 
        self.logger.info(self.prices_to_match)
        self.logger.info("estimates:")
        self.logger.info(self.estimates)      
      

      
    def __get_series(self):      
        self.__load_data()
        
        
        # assuming estimates is your array of estimate values
        # and real_value is the actual value you want to compare with
        
        mean       = list()
        std_dev    = list()
        real_value = list()
        previous_day = list()
        
        for k in self.estimates.keys():
            mean.append(self.estimates[k][0])
            std_dev.append(self.estimates[k][1])
            real_value.append(self.prices_to_match[k])
            previous_day.append(self.previous_day_price[k])
            
        
        self.logger.info(mean)
        self.logger.info(std_dev)
        self.logger.info(real_value)
        self.logger.info("-----------")
        return mean,std_dev,real_value,previous_day
    
    
    def return_quality_stats_line(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import norm
        
        mean,std_dev,real_value,previous_day =   self.__get_series()
        
        
          
        fig, axs = plt.subplots(2)

            

        axs[0].scatter(mean, mean, label='estimates', color='black', s=2)
        axs[0].scatter(mean, real_value, label='actual values', color='blue', s=2)
        
        perc = [ ((est-prev)/prev - (real-prev)/prev) for est, real, prev in zip(mean, real_value, previous_day)]
        
        axs[1].hist(x=perc, bins=100)
        mu, std = norm.fit(perc)
        norm_x = np.linspace(mu - 4*std, mu + 4*std, 100)
        p = norm.pdf(norm_x, mu, std)
        axs[1].plot(norm_x, p, 'k', linewidth=2)
        
        axs[1].axvline(mu, color='r', linestyle='--', label='Mean')
        axs[1].axvline(mu - std, color='b', linestyle='--', label='Mean - STD')
        axs[1].axvline(mu + std, color='b', linestyle='--', label='Mean + STD')
        
        # Calculate coefficients for the polynomial (change the degree as needed)
        coefficients = np.polyfit(real_value, real_value, deg=1)
        
        # Generate a polynomial function
        polynomial = np.poly1d(coefficients)
        
        # Generate x values
        x_values = np.linspace(min(real_value), max(real_value), 500)
        
        # Generate y values
        y_values = polynomial(x_values)
        # Add labels to the dots
        for i, label in enumerate(self.estimates.keys()):
            axs[0].text(mean[i], mean[i], label, color="black", fontsize=6)
            axs[0].text(mean[i], real_value[i], label, color="blue", fontsize=6)
            
        
        # Add error areas 1,2,3 sigma
        for i in range(len(mean)):
            axs[0].fill_between([mean[i] - std_dev[i], mean[i] + std_dev[i]], mean[i] - std_dev[i], mean[i]+ std_dev[i], color='darkgreen', alpha=0.2)
            
        for i in range(len(mean)):
            axs[0].fill_between([mean[i] - 2*std_dev[i], mean[i] + 2*std_dev[i]], mean[i] - 2*std_dev[i], mean[i]+ 2*std_dev[i], color='green', alpha=0.2)
            
        for i in range(len(mean)):
            axs[0].fill_between([mean[i] - 3*std_dev[i], mean[i] + 3*std_dev[i]], mean[i] - 3*std_dev[i], mean[i]+ 3*std_dev[i], color='lightgreen', alpha=0.2)
            
        
        
        axs[0].set_xlabel('R_estimate - R_real')
        axs[0].set_ylabel('Y')
        axs[0].plot(x_values, y_values, label='Fitted line', color="black", linestyle=":")
        axs[0].set_title('Actuals vs Estimates ' + " @ " + self.next_date_str)
        
        #plt.legend()
        plt.grid(True)
        #plt.tight_layout()
        plt.show()
        
        return 
        
        
        
if __name__ == '__main__':
    init_config("config.json")
    logger = logging.getLogger('quality_checker.logger')
    file_handler = logging.FileHandler('quality_checker.log')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info("Quality Checker ...")
    
    
    qc = QualityChecker(report_folder = FOLDER_REPORD_PDF, prevision_date = datetime.strptime("29-12-2023", "%d-%m-%Y"), logger = logger)
    qc.return_quality_stats_line()
    
        