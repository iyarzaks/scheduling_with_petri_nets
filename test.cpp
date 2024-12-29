#include <iostream>
#include <fstream>
#include "json.hpp"  // כולל את קובץ json.hpp

using json = nlohmann::json;  // alias ל-nlohmann::json

int main() {
    // פותחים את הקובץ לקריאה
    std::ifstream input_file("rcpspExample.json");
    
    // אם לא הצלחנו לפתוח את הקובץ
    if (!input_file.is_open()) {
        std::cerr << "לא הצלחנו לפתוח את הקובץ rcpspExample.json" << std::endl;
        return 1;
    }

    // יצירת אובייקט JSON
    json j;
    
    // קריאה לתוך האובייקט JSON
    input_file >> j;

    // הצגת תוכן ה-JSON
    std::cout << "תוכן הקובץ JSON:" << std::endl;
    std::cout << j.dump(4) << std::endl;  // dump(4) מדפיס עם אינדנטציה של 4 רווחים

    

    // סגירת הקובץ אחרי סיום הקריאה
    input_file.close();

    return 0;
}
