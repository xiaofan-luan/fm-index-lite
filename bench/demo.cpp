// Quick demo of the new anchored + case-insensitive queries.
#include <cstdio>
#include <string>
#include <vector>
#include "index/fmindex/FMIndex.h"
using namespace milvus::index::fmindex;

static const uint8_t* B(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}

int main() {
    // ---- build a tiny "log" corpus, one line = one document ----
    std::vector<std::string> docs = {
        "ERROR disk full on /dev/sda1",   // 0
        "warn: low memory",               // 1
        "error timeout talking to db",    // 2
        "erratic sensor reading",         // 3
        "request done in 12ms",           // 4
    };
    std::string concat;
    std::vector<uint64_t> starts;
    for (auto& d : docs) { starts.push_back(concat.size()); concat += d; }

    // case-insensitive index
    FMIndex fm;
    fm.Build(B(concat), concat.size(), 32, /*case_insensitive=*/true);
    fm.SetDocStarts(starts);

    printf("corpus (%zu docs):\n", docs.size());
    for (size_t i = 0; i < docs.size(); ++i) printf("  [%zu] %s\n", i, docs[i].c_str());

    auto show = [&](const char* what, const std::vector<uint64_t>& ids) {
        printf("%-42s -> docs {", what);
        for (size_t i = 0; i < ids.size(); ++i) printf("%s%llu", i ? "," : "",
                                                        (unsigned long long)ids[i]);
        printf("}\n");
    };

    printf("\n-- substring Count (case-insensitive) --\n");
    printf("Count(\"error\") = %zu   Count(\"ERROR\") = %zu   Count(\"ErRoR\") = %zu\n",
           fm.Count(B("error"), 5), fm.Count(B("ERROR"), 5), fm.Count(B("ErRoR"), 5));

    printf("\n-- anchored prefix: documents that BEGIN with ... --\n");
    show("LocatePrefixDocs(\"error\")", fm.LocatePrefixDocs(B("error"), 5));
    show("LocatePrefixDocs(\"err\")",   fm.LocatePrefixDocs(B("err"), 3));
    show("LocatePrefixDocs(\"disk\") (occurs, not at start)",
         fm.LocatePrefixDocs(B("disk"), 4));

    printf("\n-- anchored suffix: documents that END with ... --\n");
    show("LocateSuffixDocs(\"memory\")", fm.LocateSuffixDocs(B("memory"), 6));
    show("LocateSuffixDocs(\"ms\")",     fm.LocateSuffixDocs(B("ms"), 2));

    printf("\n-- Locate offsets still point into ORIGINAL text --\n");
    auto pos = fm.Locate(B("error"), 5);
    printf("Locate(\"error\") text offsets = {");
    for (size_t i = 0; i < pos.size(); ++i)
        printf("%s%llu", i ? "," : "", (unsigned long long)pos[i]);
    printf("}  (0 = \"ERROR\", %llu = \"error\")\n",
           pos.size() > 1 ? (unsigned long long)pos[1] : 0);
    return 0;
}
