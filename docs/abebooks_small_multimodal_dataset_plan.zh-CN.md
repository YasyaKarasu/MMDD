# AbeBooks 小规模多模态图书数据集调研与实现方案

## 结论

可行，但应把“可行”分成两层：技术上可以按 ISBN 自动定位当前 AbeBooks 页面、抽取书目和商品字段、保存封面/卖家图片，并由本地模型从简介和图片中抽取属性；合规上不能直接对 AbeBooks 页面做批量爬取。AbeBooks 当前条款禁止未经许可的数据挖掘、robots 和类似提取工具；公开的 Search Web Services 需要加入 Affiliate Program 并申请 Client Key，而且其条款限制缓存和再分发 Search Results。**项目负责人已于 2026-09-13 决定：在未取得 SWS 授权的情况下，先用低速率浏览器采集跑 50–100 个 ISBN 的 pilot，详见下文“风险、许可与已记录的决定”。** 取得授权仍是把数据集用于训练和发布的推荐路径。

## 现有数据与论文基线

`book.txt` 是四列 TSV：`source, isbn, title, author list`。本地统计为 33,971 行、1,265 个 ISBN、895 个 source 名称、24,819 个 `(source,isbn)` 对；作者为空 713 行，`Not Available` 23 行，且有 26,554 个完全重复行外的唯一行。论文报告的原始实验是 1,265 本书、894 个书店、34,031 条 listing，平均一本书有 5.4 组不同作者；作者真值通过随机抽取 100 本并读取书封来人工建立，TruthFinder 约在三轮迭代后达到约 95% accuracy。论文的对象是“同一本书上多个供应商对作者事实的冲突”，并不要求新数据集保留所有 listing。

本项目建议把每个 ISBN 最终只保留一条代表性记录。这样减少候选数、避免同一 ISBN 的重复商品压低召回率；原始候选若需要审计，可放在临时缓存而不进入训练表。注意 ISBN 并不总能唯一决定装帧或版本：当前页面会同时出现 hardcover、softcover、international edition、不同年份和不同卖家描述。因此选择代表记录时必须保存 `edition/binding/publication_year`，不能只按 ISBN 合并。

## 当前页面能提供的证据

对 ISBN `0201853949` 的当前页面，页面给出了 ISBN-10/13、出版社、出版年、语言、binding、edition number、页数、目录封面、简介、作者简介、封底文字、Goodreads rating，以及若干卖家商品的书况、价格、运费、库存、卖家库存号和卖家图片。`0201558025` 页面还明确显示多种装帧和 international edition；商品详情页提供 seller inventory number、书况描述和库存图片。卖家 storefront 页面可提供卖家名称、所在地、加入 AbeBooks 时间、星级、经营类别和退货/运输政策。

**2026-09-13 pilot 实测修正（42 个成功详情页）**，上段有两处需要更正：

- **页数不可得**：`page_count` 在 42 页中只有 1 页以正文形式出现「N pages」，且不存在承载它的结构化字段（`data-test-id` 中没有 `page*`）。应视为当前页面不提供。
- **封底文字不可得**：没有对应字段；"back cover" 一词只出现在卖家自述书况里（如"封底有折痕"），不能当作封底文本。
- **作者简介的取值节点**是 `data-test-id="about-the-title"`（实测非空率 95%）。同页的 `about-description` 是**卖家自我介绍**（"We are an independent online bookseller…"），不是作者简介，两者必须分开取。
- 结构化书目应以 `data-test-id="bibliographic-details-*"` 为准（`edition`/`binding`/`publisher`/`publishyear`/`language`/`condition`/`dustjacket`/`dimensions`/`itemweight`/`series`），它描述**这一本**；JSON-LD 只描述该版本，粒度更粗，宜作兜底。

其余字段实测非空率（分母为 42 个成功详情页）：

| 字段组 | 非空率 |
| --- | --- |
| title / authors / isbn10 / isbn13 / publisher / publication_year / language / binding / condition / catalogue_image_url / stock_image | 100% |
| synopsis_text / vendor_description | 98% |
| about_author_text / seller_terms / seller_description | 95–98% |
| seller_inventory_no / shipping_price / seller_rating / seller_location | 100% |
| goodreads_rating / goodreads_rating_count | 90% |
| shipping_terms | 83% |
| seller_city / seller_region / seller_country（由 `sf-address-line` 三段切分，故同为 76%） | 76% |
| edition_number | 45% |
| item_weight | 40% |
| dust_jacket | 21% |
| dimensions | 19% |
| series | 7% |

搜索卡片 12 个字段（listing_id/title/authors/seller/condition/price/currency/quantity/rating/stock_image/url）在 42 条记录上均为 98%（唯一缺口是那条零 listing 记录）。每个 ISBN 平均 12.4 条 listing、11.1 个不同卖家；**39/42 条记录的多个卖家在作者列表上互相矛盾**——这正是论文要研究的冲突信号，说明 pilot 数据足以支撑该任务设计。

**2026-09-13 扩到 200 个 ISBN 的复核**：结果 176 条 ok、17 条 no_listings、7 条 degraded（全部来自 pilot 首轮）、0 blocked、0 error。**新增的 ~150 条记录退化率为 0**，说明退化率不随累计请求量上升——这是放量前唯一未验证的风险点，当时据此认为已排除（该判断后来被 283 条处的 503 推翻，见下节）。各字段非空率与上表基本一致（所有 100% 组保持 100%；synopsis/vendor_description 98%、about_author_text 93%、shipping_terms 82%、seller_city/region/country 79%、edition_number 32%、item_weight 48%、dust_jacket 28%、dimensions 17%、series 3%）。两处变化值得注意：`goodreads_rating` 从 90% 降到 73%（样本变大后暴露出更多无 Goodreads 评分的书目），`no_listings` 从 2% 升到 8%。平均每个 ISBN 9.5 条 listing、8.4 个卖家；**148/176 条记录存在作者列表冲突**，比例与 pilot 一致。

**2026-09-13 暂停时的 253 条 ok 复核**：覆盖率在样本再扩大 1.4 倍后保持稳定，说明表结构已可定稿——`title/authors/isbn13/publisher/binding/condition/catalogue_image_url/seller_name/seller_url/seller_since/seller_location/seller_rating/seller_inventory_no/shipping_price` 均为 100%；`synopsis_text`/`vendor_description` 98%、`seller_description` 98%、`seller_terms` 97%、`about_author_text` 94%、`shipping_terms` 84%、`seller_city/region/country` 76%、`goodreads_rating` 74%、`item_weight` 52%、`edition_number` 29%、`dust_jacket` 28%、`dimensions` 18%（**此值偏高：46 个非空里 36 个是字面量 `"N/A"`，清哨兵后的真实覆盖是 4%**，见「2026-09-14 四张表落地」）、`series` 2%。搜索卡片 12 个字段仍为 100%。平均每个 ISBN **9.7 条 listing、8.6 个卖家**；**210/253 条记录（83%）的多个卖家在作者列表上互相矛盾**——冲突信号充足，足以支撑论文要研究的任务设计。`degraded` 记录仍保留完整的搜索数据（12/12 条），只有详情字段缺失。

官方 SWS 文档列出的可编程字段更适合自动化：Book ID、ISBN-10/13、listing/item condition、quantity、currency、listing/total price、shipping cost/days、listing URL、author、title、publisher、catalogue image、vendor name/location/ID/rating、seller keywords、binding、first-edition flag、dust-jacket、publication year、vendor price/description、vendor image、product type、all vendor image URLs 和 language。搜索可按 ISBN、author、title、keyword、publisher、condition、binding、seller location/rating 过滤，单次最多 200 条结果；Subjects 已在 2024 年移除。页面和 SWS 都可能把 synopsis 标成属于其他 edition，必须将其 provenance 标为“页面级简介/版本可能不确定”。

## 建议的三张主表和证据表

最终训练数据只从每个 ISBN 选择一条 `listing`，但保留可复现的来源 URL、抓取时间和证据定位。

**以下四小节已于 2026-09-14 落地并定稿**，列清单为 `output/abebooks_dataset/*.jsonl` 的实际列。相对本节最初设想的偏离逐条记录在「2026-09-14 四张表落地」一节。

### `book_edition`

一行一个规范化 ISBN 版本。**共 31 列。**

`book_id, isbn_scheme, isbn10, isbn13, title, authors, title_source, authors_source, detail_listing_seller_id, publisher, publication_year, publication_year_raw, language, binding, edition_number, series, dust_jacket, product_type, dimensions, item_weight, copy_condition_grade, condition_source, catalogue_image_url, catalogue_image_kind, synopsis_text, about_author_text, goodreads_rating, goodreads_rating_count, representative_listing_id, source_url, retrieved_at`

其中 ISBN、出版社、年份、binding 是高置信结构化字段；简介、作者简介是文本 evidence，不应未经标记地当作同一版本的事实。**`title`/`authors` 是对应代表 listing 的卡片字符串（实测逐字节相等），是卖家声明而非版本真值**——故不叫 `canonical_*`，并用 `title_source`/`authors_source`/`detail_listing_seller_id` 标出它来自哪条声明。

### `seller`

一行一个被代表 listing 引用的卖家。**共 22 列。** 同一 seller_id 若被多条记录引用，折叠为一行（取字段最丰富的一次观测），`observations` 记观测数、`fields_from_detail_page` 记哪些列来自详情页。

`seller_id, seller_name, seller_name_variants, seller_name_collision, seller_key_method, seller_url, location_raw, city, region, country, seller_rating, seller_rating_kind, seller_since, seller_since_iso, specialties, terms_of_sale, shipping_terms, seller_description, fields_from_detail_page, seller_fields_retrieved_at, seller_page_fetched, observations`

若某 ISBN 的代表记录没有 seller ID，则使用规范化 storefront URL 或受控的 `(name, location)` 键，并保留 `seller_key_method`。不要把卖家名称中的大小写和标点直接当作不同 source——`seller_name_variants` 与 `seller_name_collision` 就是为此保留的。

### `book_listing`

一行一个 ISBN 的代表商品（因此和 `book_edition` 通常是一对一，但保留独立表以表达商品层属性）。**共 27 列。**

`listing_id, book_id, seller_id, listing_url, seller_inventory_no, listing_title, listing_authors, condition, condition_description, availability_quantity, price, currency, shipping_price, shipping_currency, total_price, vendor_description, vendor_image_url, image_kind, stock_image_flag, edition_marker, selected_reason, detail_listing_id, detail_listing_ambiguous, detail_listing_candidate_count, detail_match_method, cards_truncated, retrieved_at`

`selected_reason` 取 `evidence_complete`、`seller_photo_elsewhere`、`stock_image_only`、`seller_image_preferred`、`detail_unmatched` 等受控值。选择规则应优先：ISBN/版本一致，其次结构化字段完整，再次有真实 seller image 和 description，最后才按价格或默认排序。不要因为卖家星级高就推断书目字段正确。

### `evidence_asset`（第四张物理表）

**共 19 列。**

`evidence_id, book_id, record_isbn, listing_id, is_representative_listing, asset_type, uri, local_path, sha256, mime_type, source_page, source_locator, caption_or_alt, extracted_text, text_chars, extraction_method, confidence, fetch_status, retrieved_at`

`asset_type` 取 `search_page`、`detail_page`、`degraded_page`、`catalogue_cover`、`seller_cover`、`description`、`synopsis`、`about_author`、`seller_policy`、`shipping_policy`。图片和文本证据都保留原始 URI、页面 URL、抓取时间和哈希；训练时可以把 `extracted_text` 及图片作为 text/image evidence，验证时仍可回到原页面。

**一处必须写明的口径**：文本证据行的 `sha256`/`local_path` 是**其所在页面**的，不是该段摘录的哈希（`source_locator` 才定位段落在页内的位置）。`fetch_status` 区分「未下载」与「下载失败」；图片行现在是 `url_only`。

## 2026-09-14 四张表落地：口径、实测与偏离

采集暂停期间用现有 **289 条记录**（253 ok、21 no_listings、14 degraded、1 blocked）把四张表建成，产物在 `output/abebooks_dataset/`，构建脚本 `src/build_abebooks_dataset.py` + 纯函数模块 `src/mmdd_dataset/abebooks_tables.py`，40 个测试。构建幂等、无状态，可对追加后的记录重复施加，故采集恢复后可直接增量重跑。

**规模**：`book_edition` 253、`seller` 50、`book_listing` 253、`evidence_asset` 2305；`unresolved` 10、`superseded` 5。

**入表口径**：

- **去重**：每个 ISBN 取 `max(records, key=(status_rank, retrieved_at, line_index))`，`status_rank` 为 ok=5…error=1。今天对 253 条 ok 是 no-op，作用在重跑——重跑只会追加更好的记录，规则会把 ISBN 自动提升。落选者写 `superseded_records.jsonl` 而非静默丢弃。
- **`no_listings` 不建行**，只进 `stats.json`；`degraded`/`blocked` 是可重试的**未知**而非事实，进 `unresolved.jsonl`。计划里预期的 11 条 unresolved 实测为 **10**：去重后 `0201616475` 的 blocked 记录（rank 2）输给它自己的 degraded 记录（rank 3），成为 5 条 superseded 之一。
- **代表 listing 锚定在详情页描述的那条卡片**（Stage A）。匹配路径实测 `seller_url` 252、`seller_name` 1。**URL 规范化必须剥掉 query**：`book.seller_url` 无 `?ref_`、卡片 URL 带 `?ref_=nav_sflk_srp`，不剥是 0/253 匹配，剥了是 252/253。

**选择规则的一处补充**：计划的 `selected_reason` 四个受控值无法划分全部 253 条，新增 **`seller_photo_elsewhere`**（代表卡片是详情页那张、但该 ISBN 的真实卖家照在别的卡片上），实测分布 `evidence_complete` 27 / `seller_photo_elsewhere` 126 / `stock_image_only` 100。

**`--prefer-seller-image` 对照（实测，两套标注不混用）**：

| | 详情页锚定（默认） | `--prefer-seller-image` |
| --- | --- | --- |
| `seller` 行数 | 50 | 81 |
| `book_listing.shipping_price` / `total_price` / `seller_inventory_no` | 100% | **50.2%** |
| `book_listing.vendor_description` | 98.4% | **49.4%** |
| `book_listing.listing_authors` | 100% | 99.2% |
| 真实卖家照在代表行 | 27 | 153 |

这张表就是「为什么选详情页锚定」的实测形态：换卡片让 126 条记录丢掉详情页独有的 4 列。**计划里两处估计需要更正**：`--prefer-seller-image` 下 `seller` 实测 81（原估 ≈100）；换卡片替掉作者声明的实测只有 **2 条**（原估 74 条，高估了）。丢列这一条不变，且比原先描述的更严重。

**对文档原始 schema 的偏离**（判据：有可信的将来来源（SWS）→ 保留 NULL；没有 → 删列）：

| 表 | 字段 | 处理 | 理由 |
|---|---|---|---|
| book_edition | `canonical_title`/`canonical_authors` | 改名 `title`/`authors` | 实测与所选卡片字符串完全相等，是卖家声明而非版本真值 |
| book_edition | `page_count`、`back_cover_text` | **删列** | 全文件 0 次出现，SWS 也没有 |
| book_edition | `copy_condition_grade` | 新增 | `bibliographic-details-condition`，100% 非空；不新增就会丢，因为 `book_listing.condition` 用卡片词汇表（`Used - As new` vs `As New`，两套词表并存） |
| book_edition | `dimensions`、`item_weight` | 哨兵值转 NULL | `dimensions` 46 个非空里 **36 个是字面量 `"N/A"`**，真实覆盖 **10/253 = 4%** |
| book_listing | `min_shipping_days`/`max_shipping_days` | **删列** | 从未解析；只有 `shipping_terms` 散文 → 属模型抽取产物 |
| book_listing | `vendor_keywords` | **删列** | 无此字段；属抽取表 |
| book_listing | `vendor_image_urls` → `vendor_image_url` + `image_kind` | 改名 | 只有卡片的一张图（`allvendorimageurls` 是 SWS 独有） |
| seller | `terms_of_sale`/`shipping_terms` | 填 ← `seller_terms`/`shipping_terms` | 88% / 84% 非空 |
| seller | `seller_page_retrieved_at` | 改名 `seller_fields_retrieved_at` + `seller_page_fetched=false` | storefront 页从未抓取，值来自「引入该卖家的那一张详情页」 |

**哪些现在就能填、哪些不能**：文档原文说第四张表「需要补图片下载」——**这句话不准确**。`evidence_asset` 除图片行外今日已完全填满：页面快照 `search_page` 284、`detail_page` 253、`degraded_page` 2 都带真实 `sha256`/`local_path`；文本证据 `description`/`synopsis`/`about_author`/`seller_policy`/`shipping_policy` 都是 245–249 行真实值。**只有图片行是 `url_only`**（`local_path`/`sha256`/`mime_type` 为 NULL）。另外 `asset_type` 由 URL 家族推导而非假设：253 个 `catalogue_image_url` 里 27 个其实指向详情页卡片自己的 `/inventory/` 照，归为 `seller_cover`。

**两处口径必须写死**：(1) 文本证据行的 `sha256`/`local_path` 是**所在页面**的，不是该段摘录的哈希；(2) `image_kind` 由 URL 判定而非 `stock_image` 标志——有 13 张卡片 `stock_image=False` 但 URL 是 `/isbn/`，而 URL 才是将来要下载的对象，原 flag 另存 `stock_image_flag`。

**已知的样本内异常（`stats.json` 可见）**：`detail_listing_ambiguous=true` 26 条（其中 11 条价格不同）、首页 30 张卡被截断 5 条。

## 自动化采集流程

1. **种子整理**：逐行读 `book.txt`，按 ISBN 去重；保留原始 title/author/source 的计数和冲突集合，只把唯一 ISBN 送入采集队列。建议先取 50–100 个 ISBN 做小试验集。
2. **入口**：有 Client Key 时优先使用获批准的 SWS。按 ISBN 请求 `outputsize=long`、`allvendorimageurls=yes`、`shippingdetails=yes`，设置固定 `targetsite`、`destinationcountry` 和 `sortorder`，保存 XML 原文的短期审计缓存。无 Client Key 时，按 2026-09-13 的决定使用 `src/abebooks_scraper.py` 的低速率浏览器采集，并遵守该节列出的约束（单页面、8–20 秒间隔、不反检测、被拒即停）。
3. **候选筛选**：同一 ISBN 的结果先按 ISBN-13/10、publication year、binding、edition 等硬条件分组；从匹配组中选择字段完整、至少有一段 description 或一张 vendor image 的 listing。对于国际版、不同年份、明显不同装帧，建立不同 `book_id` 或标记为 `edition_conflict`，不要静默合并。
4. **页面补充**：对选定 listing 读取允许访问的商品页和 seller storefront，解析稳定的 HTML `id`/`data-test-id`/`itemprop` 字段；页面示例存在 `publisher`、`isbn10`、`isbn13`、`edition-number`、`number-of-pages`、`itemprop=about` 和商品图片。解析器应输出字段值、页面 URL 和 locator，而非只输出扁平 CSV。
5. **图片下载**：只下载被选 listing 的 catalogue/vendor image；先记录 URL、响应 MIME、尺寸和 SHA-256，再下载到受控目录。保存 `stock_image` 标记，因为 seller image 可能是真实库存，而 stock image 可能与实际封面不一致。
6. **本地模型抽取**：规则解析先抽 ISBN、年份、页数、装帧、货币和价格；本地小模型只处理 description/synopsis/back-cover 和图片中的可见属性，输出严格 JSON：`attribute, value, evidence_id, span_or_bbox, confidence, status`。推荐抽取：主题/关键词、是否含练习或附答案、是否有插图、版本/国际版提示、封面上的标题/作者/出版社文字、书况细节、卖家是否声明 ex-library/notes/dust jacket。模型不得补全页面没有证据的值。
7. **质量控制**：用 ISBN 校验、字段类型校验、图片可解码检查、来源定位检查和跨字段一致性检查；对低置信或版本冲突项只保留候选属性并进入很小的人工复核集。人工不需要逐本访问页面，只复核模型置信度低、图片文字与结构化字段矛盾的记录。
8. **输出与划分**：训练主表按 `book_id` 去重；text/image evidence 通过 `evidence_id` 关联。按 book_id 划分 train/dev/test，避免同一 ISBN 的不同 evidence 泄漏到不同 split。发布前移除未授权的原始页面快照和不允许再分发的图片，仅保留在许可范围内的派生记录或 URI。

## 代表记录选择与多模态任务设计

每 ISBN 一条 listing 后，仍然可以构造有意义的 joinability 任务：`book_edition` 是 query/target table，`seller` 和 `evidence_asset` 是带文本/图片的 target/evidence；join key 可以是 ISBN、publisher+year、author surname 或 title token。不要把 seller 作为每本书的重复行，否则表的主键会退化为 marketplace listing，召回评测会更多地测“找同一卖家/同一商品”。

建议保留两套键：`isbn_exact` 用于确定性 join，`title_author_normalized` 用于受控的模糊 join。可从相同 ISBN 的原始候选中生成冲突属性标签（如多个作者列表、年份或 binding），但这些标签不应增加主表行数。

## 风险、许可与已记录的决定

**已知的合规事实，未变：**

- robots.txt 对普通 `User-agent: *` 禁止 `/search/`、`/servlet/` 等路径，`/servlet/SearchResults` 只对 Googlebot/bingbot/msnbot 等白名单爬虫开放；条款页面还禁止一般性 scraping/data mining。不能把"浏览器能打开"等同于"允许批量采集"。
- 官方 SWS 需 Affiliate Program 和 Client Key，是唯一有明确授权的自动化入口。截至本文档更新时尚未申请。
- 在仓的 `wdc_schemaorg_2023/Book/Book_abebooks.com_October2023.json.gz`（8,515 行 / 5,643 ISBN）是已授权的替代来源，但与 `book.txt` 的 1,265 个 ISBN 只重叠 **1 个**，无法替代。

**2026-09-13 决定：在未取得 SWS 授权的情况下，用低速率浏览器采集推进 pilot。**

背景是本项目的来源数据 `book.txt` 只覆盖上述 ISBN 中的 1 个，其余无法从已授权来源获得。决定由项目负责人做出，记录在此以便投稿和伦理审查时说明这是评估后的主动选择，而非疏漏。

决定同时附带的运行约束，`src/abebooks_scraper.py` 中已实现：

- **低速**：一次只开一个页面，记录间隔 8–20 秒随机，每 8 条插入一次 90 秒长间隔；全部可经 CLI 调整。
- **不伪装、不反检测**：使用真实 Chromium 与其自带的 user-agent；不伪造 UA 字符串、不注入 `navigator.webdriver` 补丁、不处理验证码。被识别为自动化并被拒时，接受该结果。
- **被拒即停**：`detect_block` 命中 403/429/503 或人机验证页面时抛出 `BlockedError` 并终止整轮运行，不重试、不换 IP、不降速重试。
- **先 pilot 后放量**：先跑 50–100 个 ISBN 做字段覆盖率评估，根据结果再决定是否扩展到全部 1,265 个。
- **详情页会整段退化，且这是限流信号（2026-09-13 实测）**：详情页有两种响应——完整页（约 310–800 KB，含 JSON-LD 与 React 渲染）和未渲染的精简页（约 140–190 KB，无 JSON-LD、无 `data-test-id`）。退化期间**全新、从未请求过的 URL 同样退化**，所以它不是单条 listing 的问题。**搜索页不受影响**：对 21 条 `no_listings` 记录的离线复核显示它们都是真实零结果页（含 71–246 个 `data-test-id`、标题正常、含 "0 results" 字样），不是未水合页被误判。
- **退化的真正性质是「瞬时的、站点侧的」，此前两个结论均已撤回（2026-09-13 更正）**：曾据此写过「脏 profile 会诱发退化」和「每个 URL 只被完整服务一次」，两者都来自**被混淆的对照**——重启 profile 的那次用的是全新 ISBN，而保留 profile 的那次用的是已请求过的 ISBN，差异被错误归因。当时代码把限流误判成单条 listing 的问题，连续消耗了 pilot 首轮的 7 个 ISBN。正确认识由今天两组直接测量给出：`0201616475` 的 3 个详情 URL 在 14:02:26 全部未水合，**14:15 再请求时 3 个全部返回完整页（321/296/297 KB）**；随后用 6 个外来 ISBN 按 scraper 自身节奏（12 秒间隔）连续请求，**6/6 全部水合**。据此，退化既不绑定 URL，也不绑定客户端配置，而是会在数分钟到十数分钟内自行消失的站点侧状态。
- **据此的运行时约束（已更新）**：默认**不启用持久化 profile**（需显式 `--profile` 才开启；这条作为保守默认保留，但已不再声称 profile 是诱因）；**`degraded` 不再计入 `SPENT_STATUSES`**——既然退化会自行消失，把它当作既成事实会为瞬时故障永久作废一个 ISBN，实测 10/285 就是这样丢的，rerun 现在会重试这些 ISBN；检测到某条记录详情页全部退化时先按 `--degrade-cooldown`（默认 180 秒）空转再继续，连续 `--max-consecutive-degraded`（默认 3）条退化则中止整轮并把决定交给操作者；退化时的详情页 HTML 会存一份 `{isbn}_degraded.html`，否则事后无从与「同一个 URL 后来正常」对照。
- **单次探测不足以判定站点恢复（2026-09-13）**：14:02:14 一次 2 请求的探测通过后，**12 秒后**同站 3 个全新详情页全部退化。因此守护脚本的探测改为要求**连续 3 个**详情页水合才判定恢复，连续退化的计数会被重置。
- **一次 HTTP 503 与后续的振荡（2026-09-13）**：第 284 条（`0201616475`）的搜索页在 12:58 返回 503，运行按设计停止；暂停 60 分钟并探测通过后于 14:02 恢复，随即遇到上述退化。503 与退化的关系未能确定——它们是两类不同信号，探测无法区分「站点健康」与「搜索页会被单独拒绝」。
- **留有审计线索**：每条记录保存 `source_url`、`retrieved_at` 和原始 HTML 的 `html_sha256`，可回溯到具体页面状态。

**仍然有效的停止条件：** 若 AbeBooks 以书面形式（含邮件）要求停止，或将其用于训练集被明确拒绝，则停止批量构建，改用开放书目元数据与已授权样本。定期抓取失败率显著上升应视为对方拒绝服务的信号，按停止条件处理，而不是加大规避力度。

**其余风险不变：** 商品库存、价格、卖家和图片会变化，必须记录 `retrieved_at`，不能宣称这是静态"最新版真值"；页面简介可能属于其他 edition，seller description 可能含营销文本、跨语言模板或错误 ISBN，所有抽取值必须带 `evidence_id` 和 confidence。


## 推荐的下一步

**字段覆盖评估已完成（2026-09-13，50 个 ISBN）**：42 条 ok、7 条 degraded、1 条 no_listings，逐字段非空率见上节。未做的是图片实际下载（只记录了 URL 与 stock_image 标记）和本地模型抽取置信度。

放量已启动，余下工作按优先级：

1. **放量已暂停：503 之后的失败率已经不满足继续采集的条件**。200 个 ISBN 的复核曾显示新增约 150 条记录**退化率为 0**，全量运行到 283 条时该判断仍成立。但 283 条处出现**第一次 HTTP 503 明确拒绝**，此后站点进入一个间歇性退化的状态：12:58 至 14:40 之间，详情页在「正常」与「未水合」之间以**分钟级**来回切换，累计只新增 1 条 ok。本文档「仍然有效的停止条件」一节写明「定期抓取失败率显著上升应视为对方拒绝服务的信号，按停止条件处理，而不是加大规避力度」，当前状态已落入该条，因此 14:40 起暂停采集并交由项目负责人决定，而不是继续调整节奏或重试策略。**项目负责人于 2026-09-13 15:42 决定：不停止采集，改为长冷却后探测，通过才续跑。该方案随后被证伪（2026-09-13 23:43）。** 守护脚本按指数退避实现——先等 8 小时，之后每次探测失败等待翻倍（上限 8 小时），要求**连续 5 页水合**才判定可以续跑，连续 5 轮无产出则停止。8 小时冷却后的探测确实以 **5/5 水合**通过并于 23:43:34 恢复采集，但 scraper 启动后**第一条记录就退化**、第二条退化、零产出，这已是第四次出现同一模式。当分钟内的直接对照说明了原因：好窗口真实存在（23:57 走代理的 2 条详情页全部水合），但窗口只有数十秒量级，探测能挤进去，而需要连续数小时的采集必然掉出窗口。**因此「探测通过」无法区分「站点已恢复」与「站点恰好短暂水合」，以此作为续跑闸门在原理上不成立。** 顺带排除了最后一个未测变量：直连无路由（`ERR_EMPTY_RESPONSE`），所以代理出口并非诱因。恢复采集需要一个不依赖短探测的判据，或接受在坏窗口期继续请求——后者已接近本文档禁止的「加大规避力度」，需负责人明确授权。
2. **已排除的解释（2026-09-13 实测，避免重复走弯路）**：失败既不绑定 URL（`0201616475` 的 3 个详情 URL 在 14:02 全部未水合，14:10 全部返回完整页），也不绑定客户端配置（同一段代码在前台与 `setsid nohup` 下**并发**运行，结果完全一致：两边同时退化），也与请求节奏无关（同一会话内交替使用 0 秒与 12 秒间隔，两种条件**各 4 次全部退化**）。期间三次出现「守护脚本探测连续 3 页水合通过 → scraper 启动后立刻连续退化」，说明用短探测判定站点是否可跑并不可靠。以上四条都是对随机现象的事后归因，记录在此以免再次误判。
3. **当前累计状态（2026-09-13 14:40 暂停时）**：1,265 个 ISBN 中已 settled 274 条（253 ok、21 no_listings，占 21.7%），10 条 degraded 与 1 条 blocked 都留在队列中等待重试（degraded 不再永久作废 ISBN）。恢复采集前应先确认站点恢复：用一组不在 `book.txt` 中的外来 ISBN 连续请求详情页，全部水合才值得重跑。
4. ~~**代表 listing 选择器**~~ **已于 2026-09-14 实现并在 253 条上验证**，见「2026-09-14 四张表落地」。默认口径为详情页锚定，`--prefer-seller-image` 作为可测量的对照保留。
5. ~~**四张表的 JSONL/Parquet 输出**~~ **已于 2026-09-14 落地**：`output/abebooks_dataset/` 下四张 JSONL + `stats.json` + `splits.json` + `dataset_manifest.json`。仓库无 pandas/pyarrow/polars/duckdb，故按既有约定输出 JSONL 而非 Parquet。**这一步确实不依赖继续采集**——计划中「`evidence_asset` 需要补图片下载」一句不准确，该表除图片行外今日已完全填满（见上节）。
6. **图片下载与本地模型抽取**尚未开始，按原计划推进。图片行现为 `url_only`，下载后填 `mime_type`/尺寸/`sha256` 并把 `fetch_status` 推进；抽取产物（`vendor_keywords`、自由文本书况、运输天数区间）进独立抽取表，带 `evidence_id`/span/confidence，**绝不回填到抓取列**。
7. 若 SWS 授权落地，改用官方接口替换浏览器采集，pilot 的解析器可保留用于 fixture 测试。`product_type`、`specialties`、多图 URL 等 SWS 独有的列已按「有可信将来来源即保留 NULL」的口径留在表里。

## 参考

- 论文：Yin, Han, Yu, “Truth Discovery with Multiple Conflicting Information Providers on the Web”, IEEE TKDE 20(6), 2008；论文中给出的实验规模和人工封面真值见 [PDF](https://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Trust/kdd07_xyin.pdf)。
- [AbeBooks Search Web Services 概览](https://www.abebooks.com/developer/search-web-services/overview)；[参数和返回字段](https://www.abebooks.com/developer/search-web-services/parameters)；[2025 End User Guide](https://www.abebooks.com/docs/AffiliateProgram/WebServices/end-user-guide.pdf)。
- [AbeBooks Terms and Conditions](https://www.abebooks.com/docs/legal/termsandconditions.shtml)；[SWS Terms of Use](https://www.abebooks.com/docs/affiliateprogram/webservices/terms-printable.shtml)；[robots.txt](https://www.abebooks.com/robots.txt)。
