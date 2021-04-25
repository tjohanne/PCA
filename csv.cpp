#include <fstream>
#include <iostream>
#include <sstream>   // std::stringstream
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility> // std::pair
#include <vector>

class csvInfo {
public:
  float *matrix;
  int cols;
  int rows;
  std::vector<std::string> column_names;
};

/**
 * Based on this:
 * https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
 */
csvInfo read_csv(std::string filename) {
  // Reads a CSV file into a vector of <string, vector<int>> pairs where
  // each pair represents <column name, column values>

  // Create a vector of <string, int vector> pairs to store the result
  std::vector<std::string> col_names;
  csvInfo csv;
  // Create an input filestream
  std::ifstream myFile(filename);
  int num_lines = 0;
  int num_cols = 0;
  // Make sure the file is open
  if (!myFile.is_open())
    throw std::runtime_error("Could not open file");

  // Helper vars
  std::string line, colname;
  float val;

  // Read the column names
  if (myFile.good()) {
    // Extract the first line in the file
    std::getline(myFile, line);

    // Create a stringstream from line
    std::stringstream ss(line);

    // Extract each column name
    while (std::getline(ss, colname, ',')) {

      // Initialize and add <colname, int vector> pairs to result
      col_names.push_back(colname);
    }
  }
  while (std::getline(myFile, line))
    ++num_lines;
  std::cout << "here" << std::endl;
  std::cout << "col_names.size" << col_names.size() << std::endl;
  std::cout << "num_lines" << num_lines << std::endl;
  num_cols = col_names.size();
  myFile.clear();
  myFile.seekg(0);
  std::getline(myFile, line);
  float *matrix = new float[num_cols * num_lines];
  // Read data, line by line
  int index = 0;
  while (std::getline(myFile, line)) {
    // Create a stringstream of the current line
    std::stringstream ss(line);

    // Extract each integer
    while (ss >> val) {
      matrix[index] = val;

      // If the next token is a comma, ignore it and move on
      if (ss.peek() == ',')
        ss.ignore();

      // Increment the column index
      index++;
    }
  }

  // Close file
  myFile.close();
  csv.matrix = matrix;
  csv.cols = num_cols;
  csv.rows = num_lines;
  csv.column_names = col_names;
  return csv;
}

void print_csv(csvInfo csv) {
  float *matrix = csv.matrix;
  int rows = csv.rows;
  int cols = csv.cols;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << matrix[i * cols + j];
      if (j != cols - 1)
        std::cout << ",";
    }
    std::cout << std::endl;
  }
  std::cout << "first " << csv.matrix[0] << std::endl;
  std::cout << "last " << csv.matrix[(csv.rows - 1) * csv.cols + csv.rows - 1]
            << std::endl;
  std::cout << "cols " << csv.cols << std::endl;
  std::cout << "rows  " << csv.rows << std::endl;
  std::cout << "first col name " << csv.column_names[0] << std::endl;
}