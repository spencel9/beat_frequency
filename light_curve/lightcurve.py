import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from astropy.timeseries import LombScargle
import math
import os
#%matplotlib inline

def main():
    #this file objects
    GettingData = GatheringData()
    GettingNumberOfSectors = NumberOfSector()
    ResultTable = GettingResultTable()
    DivideData = DividingDataInTwo()
    SplittingData = DividingDataInTwo()
    
    from .Results1 import removeLargeStdDev
    from .Results1 import CalcPptFromFlux
    from .Results1 import splineRemoveAndFitting
    from .Results1 import figures

    Stdev1 = removeLargeStdDev()
    calc1 = CalcPptFromFlux()
    splineRemove1 = splineRemoveAndFitting()
    figures1 = figures()
    
    from .Results2 import LargeStdev2
    from .Results2 import RemovingBadPoints
    from .Results2 import CalcPptFromFlux2
    from .Results2 import SplineRemoveAndFitting2
    from .Results2 import figures2
    
    Stdev2 = LargeStdev2()
    BadPoints = RemovingBadPoints()
    Calc2 = CalcPptFromFlux2()
    splineRemove2 = SplineRemoveAndFitting2()
    figures2 = figures2()
    Results1And2 = Combining_1_and_2()
    
    from .lombScargle import Periodogram
    from .lombScargle import PeriodogramCertTimeRange
    
    csvObj = MakingCSVToBeDivided()
    periodObj = Periodogram()
    periodCertTimeObj = PeriodogramCertTimeRange()

    #objects for divide_data
    from DivideData.divide_data import FindingFileAndDivide
    fileAndDivideObj = FindingFileAndDivide()

    #objects for Fitting_data
    from FittingData.Fitting_data import preAmpCalc
    from FittingData.Fitting_data import AmpPhaseCalc
    from FittingData.Fitting_data import plotting
    preCalc = preAmpCalc()
    AmpCalc = AmpPhaseCalc()
    plotsObj = plotting()

    # #the running of the code
    Number = int(input('Enter the Target Number only: '))
    TICNumber = 'TIC' + str(Number)
    try:
        specific_sec = int(input('Any sector in specific (just the number)? If not enter n: '))
    except ValueError:
        specific_sec = 'n'
    if(specific_sec == 'n'):
        DownloadableFiles = GettingData.SearchResult(TICNumber, ExposureTime = 20) ###To code spec. sector: -->SearchResult(sector = 60)
        lc_collection = GettingData.DownloadingData(DownloadableFiles)
        SecNum = GettingNumberOfSectors.SectorCalc(lc_collection) + 1
        print('Number of sectors: ' + str(SecNum - 1))
    else:
        DownloadableFiles = GettingData.SearchResultSector(TICNumber, ExposureTime = 20, sector = specific_sec)
    
    ###############################ONLY FOR FAKE DATA
    #lc_collection = pd.read_csv('./Lightcurve/trial200_new.csv')
    #SecNum = 1
    ##TICNumber = 'Fake Data New' ####For fake data gen
    ################################

    #############ISSUE IS ADDING IN RESULT 1 AND 2 .PY FOR PPT

    result = ResultTable.MakingTable(lc_collection)
    #result = lc_collection ##################only fake data
    ResultTable.SectorPlot(SecNum, lc_collection, TICNumber)
    for n in range (SecNum - 1):
        bounds = DivideData.GettingBounds(n, lc_collection[n]) 
        #bounds = DivideData.GettingBounds(n, lc_collection) 
        
        sector = lc_collection[n].sector
        results = SplittingData.SplittingInHalf(bounds, n, result, lc_collection, TICNumber, sector)

        #sector = 'No Sector3'
        #sectorNumber = lc_collection[0].sector
        #result1 = result  #Stdev1.stdevremove(n, result[0], TICNumber, sector)
        result_ppt_1 = calc1.calc(n, results[0], TICNumber, lc_collection)   #result1
        
        fittingAns = input("Do you want spline fitting (Y/N)? ")
        
        # while (fittingAns != 'y' and fittingAns != 'Y' and fittingAns != 'n' and fittingAns != 'N'):
        #     fittingAns = input('Entered value was incorrect, please try again: ')
        # if (fittingAns == 'y'):
        #     result3 = splineRemove1.removeAndFit(result2, TICNumber, sector)
        #     figures1.fittingAndPlotting(n,result3, TICNumber, sector)
        #     result4 = Stdev2.calcstdev(n, result[1], TICNumber, sector)
        #     result5 = result4 #BadPoints.removing(result4)
        #     result6 = Calc2.calculation(n, result5, TICNumber, sector)
        #     result7 = result6 #splineRemove2.removeAndFit(result6, TICNumber, sector)
        #     figures2.fittingAndPlotting(n, result7, TICNumber, sector)

        ##################################################################Result 1 and 2 .py
        #figures1.fittingAndPlotting(n,result2, TICNumber, sector)
        #result4 = Stdev2.calcstdev(n, result[1], TICNumber, sector)
        # result5 = result4 #BadPoints.removing(result4)
        result_ppt_2 = Calc2.calculation(n, results[1], TICNumber, sector)
        # figures2.fittingAndPlotting(n, result6, TICNumber, sector)
        combinedResults = Results1And2.combiningResults(result_ppt_1, result_ppt_2)    #Results1And2.combiningResults(results[0], results[1])
        ####################################################################

        #allSectors = result   #combinedResults
        #combinedResults = result ###for check
        if (n>0):
            char = input('Would you like to use the frequency from first sector (y/n)?')
            while (char != 'y' and char != 'Y' and char != 'n' and char != 'N'):
                char = input('Entered value was incorrect, please try again: ')


            if(char == 'y' or char == 'Y'):
                file_a = './TIC_' + str(TICNumber) + '_sec' + 'no_sec' + '_information' +'.csv'
                result = pd.read_csv(file_a, skiprows=1, names=['Frequency (1/days)', 'Frequency (cycles/day)', 'Frequency error', 'Period (days)', 'Period uncertainty', 'Day Division'])
                
                #####################################################################################################
                
                best_freq_1=result['Frequency (1/days)'].iloc[0]
                DayDivision = result['Day Division'].iloc[0]
                
                csv_filename = csvObj.makingCSV(n, TICNumber, sector, combinedResults, fittingAns)
                    
                fileAndDivideObj.findfile(csv_filename)
                fileAndDivideObj.DivideWithFile(n, csv_filename, DayDivision, TICNumber, sector, best_freq_1,best_freq_2)
                    
                AmpCalc.calc(TICNumber, sector, best_freq_1, result, best_freq_2, best_amplitude2, best_phase2, range_max) ####Will be error b/c no best_something_2 values
                    
                plotsObj.gettingPlots(n, TICNumber, sector, best_freq_1, DayDivision, result, best_freq_2, best_amplitude2, best_phase2)  


            else:
                best_freq_1, fwhm = periodObj.periodogram(n, combinedResults, TICNumber, sector)
                DayDivision, best_freq_2, best_amplitude2, best_phase2 = periodCertTimeObj.periodogram(combinedResults, best_freq_1, bounds, TICNumber,sector)
                        
                csv_filename = csvObj.makingCSV(n, TICNumber, sector, combinedResults, fittingAns)
                        
                fileAndDivideObj.findfile(csv_filename)
                range_max = fileAndDivideObj.DivideWithFile(n, csv_filename, DayDivision, TICNumber, sector, best_freq_1, best_freq_2)
                        
                AmpCalc.calc(TICNumber, sector, best_freq_1, result, best_freq_2, best_amplitude2, best_phase2, range_max)
                        
                plotsObj.gettingPlots(n, TICNumber, sector, best_freq_1, DayDivision, result, best_freq_2, best_amplitude2, best_phase2)
        else:
            best_freq_1, fwhm = periodObj.periodogram(n, combinedResults, TICNumber, sector)
            DayDivision, best_freq_2, best_amplitude2, best_phase2 = periodCertTimeObj.periodogram(combinedResults, best_freq_1, bounds, TICNumber,sector)
                        
            csv_filename = csvObj.makingCSV(n, TICNumber, sector, combinedResults, fittingAns)
                        
            fileAndDivideObj.findfile(csv_filename)
            range_max = fileAndDivideObj.DivideWithFile(n, csv_filename, DayDivision, TICNumber, sector, best_freq_1, best_freq_2)
                        
            AmpCalc.calc(TICNumber, sector, best_freq_1, result, best_freq_2, best_amplitude2, best_phase2, range_max)
                        
            plotsObj.gettingPlots(n, TICNumber, sector, best_freq_1, DayDivision, result, best_freq_2, best_amplitude2, best_phase2)
        #put something to track all of the info here, but need to rerun periodogram to get frequency of all sectors combined
       
    print('Program complete')

class GatheringData:
    
    IDNumber = 2
    Target = 'TIC' + str(IDNumber)
    TICNumber = 'TIC_' + str(IDNumber)
    ExposureTime = 20
    lc_collection = [0,0,0]

    def SearchResult(self, Target,  ExposureTime):
        ExposureTime = 20
        DownloadableFiles = lk.search_lightcurve(target = Target, exptime = ExposureTime)
        print(DownloadableFiles)
        return DownloadableFiles
    def SearchResultSector(self, Target,  ExposureTime, sector):
        ExposureTime = 20
        DownloadableFiles = lk.search_lightcurve(target = Target, exptime = ExposureTime, sector = sector)
        print(DownloadableFiles)
        return DownloadableFiles

    
    def DownloadingData(self, Downloadablefiles):
        lc_collection = Downloadablefiles.download_all()
        return lc_collection

class NumberOfSector:
    GatheringDataObj = GatheringData()
    def SectorCalc(self, lc_collection):
        NumberOfSec = len(lc_collection)
        return NumberOfSec

class GettingResultTable:

    def __init__(self):
        self.DataForSector = GatheringData()

        self.Lc_all = None
        self.Result = None
    
    def MakingTable(self, lc_collection):
        lc_all = lc_collection.stitch() ####################
        Result = lc_all.to_pandas()
        Result = Result.reset_index()
        print()
        return Result
    
    def SectorPlot(self, SecNum, lc_collection, TICNumber):
        for n in range ((SecNum - 1)):
            lc_Sector = lc_collection[n]
            lc_Sector.plot(title = './Lightcurve/aerror.png')
            save_dir = './SavedFigs'
            os.makedirs(save_dir, exist_ok=True)
            fileName = 'Sec_' + str(lc_collection[n].sector) + '_' + str(TICNumber)
            plt.savefig(os.path.join(save_dir, fileName))
        
class DividingDataInTwo:

    def __init__(self):
        self.DataObj = GettingResultTable()
        self.DataObj2 = GatheringData()

        self.Start = None
        self.Finish = None
        self.TimeWithoutFlux = None
        self.Times = []
        self.result1 = []
        self.result2 = []

    def GettingBounds(self, n, lc_collection):
        
        #########################################For artificial data
        # Start = lc_collection['time'].min()
        # Finish = lc_collection['time'].max()
        # print('Start: ' + str(Start))
        # print('Finsih: ' + str(Finish))
        ########################################

        Start = lc_collection.time[0]
        Finish = lc_collection.time[-1]

        TimeWithoutFlux = ((Finish - Start)/2) + Start
        
        Times = [Start,TimeWithoutFlux,Finish]
        
        return Times

    def SplittingInHalf(self, bounds, n, result, lc_collection, TICNumber, sector):
        self.resultObj = GettingResultTable()
        self.TICObj = GatheringData()
        
        a = bounds[0].btjd
        b = bounds[1].btjd
        c = bounds[2].btjd

        ######################fake data only
        # a = bounds[0]
        # b = bounds[1]
        # c = bounds[2]
        #########################
        
        result1 = result[(result['time']>=a) & (result['time']<=b)] 
        result1.plot.scatter(x='time', y = 'pdcsap_flux', title = TICNumber + ' raw result 1 (first half) ' + str(sector))
        save_dir = './SavedFigs'
        fileName = str(TICNumber) +'_'+ str(lc_collection[n].sector) + ' raw_results_first_half' ######Replace "No sector" with str(lc_collection[n].sector)
        plt.savefig(os.path.join(save_dir, fileName))
       
        result2 = result[(result['time']>= b) & (result['time']<= c)]
        result2.plot.scatter(x='time', y = 'pdcsap_flux', title = TICNumber + ' raw result 2 (second half) ' + str(sector))
        fileName = str(TICNumber) +'_'+ str(lc_collection[n].sector) + ' raw_results_second_half'
        plt.savefig(os.path.join(save_dir, fileName))
        
        results = [result1, result2]
        return results

# add frequency table into new file
class Combining_1_and_2:
    def combiningResults(self, result1, result2):
        frames = [result1, result2]
        result =  pd.concat(frames)
        results = result
        return results

class MakingCSVToBeDivided:
    TICNum = GatheringData()
    resultObj = GettingResultTable()
    result12 = Combining_1_and_2()

    def makingCSV(self, n, TICNumber, sector, result, fittingAns):
        save_dir = './MainSectors/'
        os.makedirs(save_dir, exist_ok=True)
        csv_filename = './MainSectors/' + str(TICNumber)+'_'+str(sector)+'.csv'
        txt_filename = './MainSectors/' + str(TICNumber)+'_'+str(sector) +'.txt'
        print(csv_filename)
        print(txt_filename)
        if (fittingAns == 'Y'):
            result.to_csv(csv_filename, index = False, header = False, columns = ['time','pdcsap_flux']) ##Was FLUX_ppt_fit_removed
            with open(csv_filename, 'r') as inp, open(txt_filename, 'w') as out:
                for line in inp:
                    line = line.replace(',', '\t')
                    out.write(line)
            return csv_filename
        else:
            result.to_csv(csv_filename, index = False, header = False, columns = ['time','pdcsap_flux']) ##was ppt
            with open(csv_filename, 'r') as inp, open(txt_filename, 'w') as out:
                for line in inp:
                    line = line.replace(',', '\t')
                    out.write(line)
            return csv_filename 