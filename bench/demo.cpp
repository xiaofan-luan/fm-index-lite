// Quick demo of the document-scoped, anchored + case-insensitive queries.
#include <cstdio>
#include <string>
#include <string_view>
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

    // Feed the split documents directly; the index injects a '\0' separator
    // between them, so every result is document-scoped. Case-insensitive here.
    FMIndex fm;
    std::vector<std::string_view> views(docs.begin(), docs.end());
    fm.Build(views, 32, /*case_insensitive=*/true);

    printf("corpus (%zu docs):\n", docs.size());
    for (size_t i = 0; i < docs.size(); ++i) printf("  [%zu] %s\n", i, docs[i].c_str());

    auto show = [&](const char* what, const std::vector<uint64_t>& ids) {
        printf("%-42s -> docs {", what);
        for (size_t i = 0; i < ids.size(); ++i) printf("%s%llu", i ? "," : "",
                                                        (unsigned long long)ids[i]);
        printf("}\n");
    };

    printf("\n-- substring Count (case-insensitive, per-document) --\n");
    printf("Count(\"error\") = %zu   Count(\"ERROR\") = %zu   Count(\"ErRoR\") = %zu\n",
           fm.Count(B("error"), 5), fm.Count(B("ERROR"), 5), fm.Count(B("ErRoR"), 5));

    printf("\n-- documents that CONTAIN ... (LIKE '%%x%%') --\n");
    show("MatchingDocs(\"error\")", fm.MatchingDocs(B("error"), 5));
    show("FuzzyMatchingDocs(\"eror\", k=1)", fm.FuzzyMatchingDocs(B("eror"), 4, 1));

    printf("\n-- anchored prefix: documents that BEGIN with ... --\n");
    show("LocatePrefixDocs(\"error\")", fm.LocatePrefixDocs(B("error"), 5));
    show("LocatePrefixDocs(\"err\")",   fm.LocatePrefixDocs(B("err"), 3));
    show("LocatePrefixDocs(\"disk\") (occurs, not at start)",
         fm.LocatePrefixDocs(B("disk"), 4));

    printf("\n-- anchored suffix: documents that END with ... --\n");
    show("LocateSuffixDocs(\"memory\")", fm.LocateSuffixDocs(B("memory"), 6));
    show("LocateSuffixDocs(\"ms\")",     fm.LocateSuffixDocs(B("ms"), 2));

    printf("\n-- LocateDocs: (doc, offset) of every occurrence --\n");
    auto hits = fm.LocateDocs(B("error"), 5);
    for (auto [doc, off] : hits)
        printf("  \"error\" in doc %llu at offset %llu\n",
               (unsigned long long)doc, (unsigned long long)off);

    printf("\n-- Extract: rebuild context around a match, within its document --\n");
    for (auto [doc, off] : hits) {
        uint64_t start = off >= 6 ? off - 6 : 0;
        std::string ctx = fm.Extract(doc, start, 20);
        printf("  doc %llu @%2llu -> \"...%s...\"\n", (unsigned long long)doc,
               (unsigned long long)off, ctx.c_str());
    }
    printf("  (letters are lowercased here because the index is "
           "case-insensitive)\n");
    return 0;
}
