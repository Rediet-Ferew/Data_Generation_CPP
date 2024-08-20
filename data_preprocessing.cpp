
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>


enum IrisCategory {
    IRIS_SETOSA,
    IRIS_VIRGINICA,
    IRIS_VERSICOLOR,
    IRIS_UNKNOWN
};


std::vector<std::vector<float>> Read_Iris_Dataset(void)
{

    std::ifstream myfile("iris.data");
  std::string line;
  std::vector<std::vector<float>> Iris_Dataset;
  std::vector<float> temp_sepal_len;
  std::vector<float> temp_sepal_wid;
  std::vector<float> temp_petal_len;
  std::vector<float> temp_petal_wid;
  std::vector<float> temp_iris_class;

  float sepal_len_f,sepal_wid_f,petal_len_f,petal_wid_f;
  float iris_class_f;

  std::string temp_string;
   int count =0;
   if (myfile.is_open())
  {
     std::cout<< "file opened successfully"<<std::endl;
      while (std::getline(myfile, line)) {
         std::replace(line.begin(), line.end(), '-', '_');
         std::replace(line.begin(), line.end(), ',', ' ');

         std::istringstream iss(line);
         count++;

         iss >> sepal_len_f>>sepal_wid_f >> petal_len_f >>petal_wid_f >> temp_string;
         temp_sepal_len.push_back(sepal_len_f);
         temp_sepal_wid.push_back(sepal_wid_f);
         temp_petal_len.push_back(petal_len_f);
         temp_petal_wid.push_back(petal_wid_f);
         if(temp_string.compare("Iris_setosa") == 0)
         {
            iris_class_f = IRIS_SETOSA;
         }
         else if (temp_string.compare("Iris_versicolor") == 0)
         {
            iris_class_f = IRIS_VERSICOLOR;
         }
         else if (temp_string.compare("Iris_virginica") == 0)
         {
            iris_class_f = IRIS_VIRGINICA;
         }else
         {
            iris_class_f = IRIS_UNKNOWN;
         }
         temp_iris_class.push_back(iris_class_f);
      }
      Iris_Dataset.push_back(temp_sepal_len);
      Iris_Dataset.push_back(temp_sepal_wid);
      Iris_Dataset.push_back(temp_petal_len);
      Iris_Dataset.push_back(temp_petal_wid);
      Iris_Dataset.push_back(temp_iris_class);
  }
  else
  {
     std::cout << "Unable to open file";
  }
  return Iris_Dataset;
}
