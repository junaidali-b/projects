#Reading CSV file from Windows D Drive containing a list of all installed
#packages in the Windows installation of R Studio
packages = read.csv("/home/junaidali_b/R/Files/Installed Packages.csv")

#Batch 1 of downloading and installing packages
#R Studio will attempt to install the entire list of libraries in the CSV file
packages_b1 = packages$Package
install.packages(c(packages_b1), dependencies = T)

#In some systems (such as mine), there is a possibility that the computer will
#get soft bricked due to the RAM being overloaded. This is more likely if there
#are many other programs running in the computer apart from R Studio at the
#time.

#The next couple of steps check all packages successfully installed in the first
#batch, and attempt to install the rest. This step may need to be repeated if
#there is a second system crash.
installed_pack = data.frame(installed.packages())$Package
packages_b2 = packages_b1[!(packages_b1 %in% installed_pack)]
install.packages(c(packages_b2), dependencies = T)

#There will be a few packages left out which are either not available on CRAN
#or simply show some error despite begin present on CRAN. 

#The ones that are available on individual Git repositories will have to be 
#downloaded using the following syntax
install.packages("remotes")
#Replace "mbtyers/jagsplot" below with desired package name from desired
#repository
remotes::install_github("mbtyers/jagsplot")



#Manual Methods

#1: Using Terminal

#Each package may be available on Arch Linux AUR (or public repositories for
#other distributions). Using the name of the package used on AUR, execute the
#following command in the terminal. Note that this syntax is only valid for the
#'yay' AUR helper on Arch Linux and other Arch based distributions, and the
#syntax for other distributions will vary. Some dependencies may also need to be
#installed using this method.

#Syntax (Terminal): yay -S r-r2jags


#2: Using Git Repositories

#This method works for libraries which have been delisted from CRAN but may
#still be available on Git

devtools::install_github("dustinfife/flexplot")


#3: Using individually downloaded .tar.gz files for installation

#For files that are available online in .tar.gz file format, the following
#syntax would work. This code takes all files in a folder (expecting all)
#files in the folder to be .tar.gz files), and installs them.

package_folder = "/home/junaidali_b/R/Packages/" #replace with relevant folder path
targz_package_list = list.files(package_folder, full.names = T)
for (i in targz_package_list){
  #Installing package from .tar.gz file
  install.packages(i, dependencies = T)
}

#Packages with 'non-zero' exit status were not installed, and in if this happens,
#it means that there were dependencies that could not be fulfilled due to
#compatibility issues.