// Chatbot (C++ version) - ICT project.
// Same idea as the Python version: TF-IDF + cosine similarity over data/knowledge.txt.
// Build:  g++ -std=c++17 -O2 chatbot.cpp -o chatbot
// Run:    ./chatbot ../data/knowledge.txt
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

using Vec = map<string, double>;

const set<string> STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "what", "how",
                          "do", "does", "i", "me", "you", "it", "for", "on", "about", "tell", "explain"};

vector<string> tokenize(const string &text) {
    vector<string> out;
    string word;
    auto flush = [&]() {
        if (!word.empty() && !STOP.count(word)) out.push_back(word);
        word.clear();
    };
    for (char c : text) {
        if (isalnum((unsigned char)c) || c == '+') word += (char)tolower(c);
        else flush();
    }
    flush();
    return out;
}

class Chatbot {
    vector<pair<string, string>> entries;
    vector<vector<string>> docs;
    map<string, double> idf;
    vector<Vec> vectors;

    Vec vectorize(const vector<string> &tokens) const {
        Vec v;
        for (const auto &w : tokens) {
            auto it = idf.find(w);
            if (it != idf.end()) v[w] += it->second;
        }
        return v;
    }

    static double cosine(const Vec &a, const Vec &b) {
        double dot = 0, na = 0, nb = 0;
        for (const auto &[w, x] : a) {
            na += x * x;
            auto it = b.find(w);
            if (it != b.end()) dot += x * it->second;
        }
        for (const auto &[w, y] : b) nb += y * y;
        return (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    }

public:
    bool load(const string &path) {
        ifstream file(path);
        if (!file) return false;
        string line;
        while (getline(file, line)) {
            size_t bar = line.find('|');
            if (bar == string::npos) continue;
            entries.push_back({line.substr(0, bar), line.substr(bar + 1)});
        }
        for (const auto &[q, a] : entries) docs.push_back(tokenize(q + " " + a));
        map<string, int> df;
        for (const auto &d : docs)
            for (const auto &w : set<string>(d.begin(), d.end())) df[w]++;
        double n = docs.size();
        for (const auto &[w, c] : df) idf[w] = log((1 + n) / (1 + c)) + 1;
        for (const auto &d : docs) vectors.push_back(vectorize(d));
        return !entries.empty();
    }

    string reply(const string &question) const {
        Vec qv = vectorize(tokenize(question));
        double best = 0;
        int bestIdx = -1;
        for (size_t i = 0; i < vectors.size(); i++) {
            double s = cosine(qv, vectors[i]);
            if (s > best) { best = s; bestIdx = (int)i; }
        }
        if (bestIdx < 0 || best < 0.12)
            return "Sorry, I don't know that yet. Try asking about AI, Python, C++, Git or hackathons.";
        return entries[bestIdx].second;
    }
};

int main(int argc, char *argv[]) {
    string path = argc > 1 ? argv[1] : "../data/knowledge.txt";
    Chatbot bot;
    if (!bot.load(path)) {
        cerr << "Could not load knowledge base: " << path << "\n";
        return 1;
    }
    cout << "Chatbot (C++) ready. Type 'quit' to exit.\n";
    string input;
    while (true) {
        cout << "\nYou: ";
        if (!getline(cin, input) || input == "quit" || input == "exit") break;
        if (input.empty()) continue;
        cout << "Bot: " << bot.reply(input) << "\n";
    }
    return 0;
}
