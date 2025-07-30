// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

// The current database to use.
use("nlp");

// db.article.dropIndex("content_text")
// db.article.createIndex(
//   { content: "text" },
//   {
//     name: "content_text",
//     default_language: "none",
//     language_override: "language"
//   }
// )

// db.article.getIndexes();

// db.article.find({ $text: { $search: "tai nạn giao thông" } })
// db.article.find({ $text: { $search: "\"tai nạn giao thông\"" } })

// db.article.find(
//   { $text: { $search: "tai nạn giao thông" } },
//   { score: { $meta: "textScore" } }
// ).sort({ score: { $meta: "textScore" } }).limit(10);


// Tạo idx 2 fields title và content
db.article.createIndex(
  { title: "text", content: "text" },
  { default_language: "none" }
)