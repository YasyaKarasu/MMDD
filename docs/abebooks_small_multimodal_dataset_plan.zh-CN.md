# AbeBooks 小规模多模态图书数据集调研与实现方案

## 结论

可行，但应把“可行”分成两层：技术上可以按 ISBN 自动定位当前 AbeBooks 页面、抽取书目和商品字段、保存封面/卖家图片，并由本地模型从简介和图片中抽取属性；合规上不能直接对 AbeBooks 页面做批量爬取。AbeBooks 当前条款禁止未经许可的数据挖掘、robots 和类似提取工具；公开的 Search Web Services 需要加入 Affiliate Program 并申请 Client Key，而且其条款限制缓存和再分发 Search Results。**项目负责人已于 2026-09-13 决定：在未取得 SWS 授权的情况下，先用低速率浏览器采集跑 50–100 个 ISBN 的 pilot，详见下文“风险、许可与已记录的决定”。** 取得授权仍是把数据集用于训练和发布的推荐路径。

## 现有数据与论文基线

`book.txt` 是四列 TSV：`source, isbn, title, author list`。本地统计为 33,971 行、1,265 个 ISBN、895 个 source 名称、24,819 个 `(source,isbn)` 对；作者为空 713 行，`Not Available` 23 行，且有 26,554 个完全重复行外的唯一行。论文报告的原始实验是 1,265 本书、894 个书店、34,031 条 listing，平均一本书有 5.4 组不同作者；作者真值通过随机抽取 100 本并读取书封来人工建立，TruthFinder 约在三轮迭代后达到约 95% accuracy。论文的对象是“同一本书上多个供应商对作者事实的冲突”，并不要求新数据集保留所有 listing。

本项目建议把每个 ISBN 最终只保留一条代表性记录。这样减少候选数、避免同一 ISBN 的重复商品压低召回率；原始候选若需要审计，可放在临时缓存而不进入训练表。注意 ISBN 并不总能唯一决定装帧或版本：当前页面会同时出现 hardcover、softcover、international edition、不同年份和不同卖家描述。因此选择代表记录时必须保存 `edition/binding/publication_year`，不能只按 ISBN 合并。

## 当前页面能提供的证据

对 ISBN `0201853949` 的当前页面，页面给出了 ISBN-10/13、出版社、出版年、语言、binding、edition number、页数、目录封面、简介、作者简介、封底文字、Goodreads rating，以及若干卖家商品的书况、价格、运费、库存、卖家库存号和卖家图片。`0201558025` 页面还明确显示多种装帧和 international edition；商品详情页提供 seller inventory number、书况描述和库存图片。卖家 storefront 页面可提供卖家名称、所在地、加入 AbeBooks 时间、星级、经营类别和退货/运输政策。

官方 SWS 文档列出的可编程字段更适合自动化：Book ID、ISBN-10/13、listing/item condition、quantity、currency、listing/total price、shipping cost/days、listing URL、author、title、publisher、catalogue image、vendor name/location/ID/rating、seller keywords、binding、first-edition flag、dust-jacket、publication year、vendor price/description、vendor image、product type、all vendor image URLs 和 language。搜索可按 ISBN、author、title、keyword、publisher、condition、binding、seller location/rating 过滤，单次最多 200 条结果；Subjects 已在 2024 年移除。页面和 SWS 都可能把 synopsis 标成属于其他 edition，必须将其 provenance 标为“页面级简介/版本可能不确定”。

## 建议的三张主表和证据表

最终训练数据只从每个 ISBN 选择一条 `listing`，但保留可复现的来源 URL、抓取时间和证据定位。

### `book_edition`

一行一个规范化 ISBN 版本。

`book_id, isbn10, isbn13, canonical_title, canonical_authors, publisher, publication_year, language, binding, edition_number, page_count, dust_jacket, product_type, catalogue_image_url, synopsis_text, about_author_text, back_cover_text, goodreads_rating, goodreads_rating_count, source_url, retrieved_at`

其中 ISBN、出版社、年份、binding、页数等是高置信结构化字段；简介、作者简介、封底是文本 evidence，不应未经标记地当作同一版本的事实。

### `seller`

一行一个被最终代表 listing 引用的卖家。

`seller_id, seller_name, seller_url, city, region, country, seller_rating, seller_since, specialties, terms_of_sale, shipping_terms, seller_page_retrieved_at`

若某 ISBN 的代表记录没有 seller ID，则使用规范化 storefront URL 或受控的 `(name, location)` 键，并保留 `seller_key_method`。不要把卖家名称中的大小写和标点直接当作不同 source。

### `book_listing`

一行一个 ISBN 的代表商品（因此和 `book_edition` 通常是一对一，但保留独立表以表达商品层属性）。

`listing_id, book_id, seller_id, listing_url, seller_inventory_no, listing_title, listing_authors, condition, condition_description, availability_quantity, price, currency, shipping_price, total_price, min_shipping_days, max_shipping_days, vendor_description, vendor_keywords, vendor_image_urls, selected_reason, retrieved_at`

`selected_reason` 建议取 `evidence_complete`、`stock_image_only`、`no_valid_listing` 等受控值。选择规则应优先：ISBN/版本一致，其次结构化字段完整，再次有真实 seller image 和 description，最后才按价格或默认排序。不要因为卖家星级高就推断书目字段正确。

### `evidence_asset`（建议保留，作为第四张物理表）

`evidence_id, book_id, listing_id, asset_type, uri, local_path, sha256, mime_type, source_page, source_locator, caption_or_alt, extracted_text, extraction_method, confidence, retrieved_at`

`asset_type` 可为 `catalogue_cover`、`seller_cover`、`description`、`synopsis`、`about_author`、`back_cover`、`seller_policy`。图片和文本证据都应保留原始 URI、页面 URL、抓取时间和哈希；训练时可以把 `extracted_text` 及图片作为 text/image evidence，验证时仍可回到原页面。

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
- **详情页会整段退化，且这是限流信号（2026-09-13 实测）**：详情页有两种响应——完整页（约 310–800 KB，含 JSON-LD 与 React 渲染）和未渲染的精简页（约 140–190 KB，无 JSON-LD、无 `data-test-id`）。精简页与客户端配置无稳定关系：同一套代码、同一个 session 内，既出现过连续正常（316/320/314 KB），也出现过连续退化；退化期间**全新、从未请求过的 URL 同样退化**，而同一 session 里一个不在 `book.txt` 中的外来 ISBN 却正常返回完整页。**搜索页不受影响**，始终返回完整结果列表。据此的运行时约束：默认**不启用持久化 profile**（需显式 `--profile` 才开启；实测脏 profile 会诱发退化）；检测到某条记录详情页全部退化时视为被限流，先按 `--degrade-cooldown`（默认 180 秒）空转再继续，而不是继续消耗后续 ISBN；已标 `degraded` 的 ISBN 计入 `SPENT_STATUSES`，rerun 不会重复请求。注意 pilot 首轮 7 条全部退化，即因未识别该信号而连续消耗。
- **留有审计线索**：每条记录保存 `source_url`、`retrieved_at` 和原始 HTML 的 `html_sha256`，可回溯到具体页面状态。

**仍然有效的停止条件：** 若 AbeBooks 以书面形式（含邮件）要求停止，或将其用于训练集被明确拒绝，则停止批量构建，改用开放书目元数据与已授权样本。定期抓取失败率显著上升应视为对方拒绝服务的信号，按停止条件处理，而不是加大规避力度。

**其余风险不变：** 商品库存、价格、卖家和图片会变化，必须记录 `retrieved_at`，不能宣称这是静态"最新版真值"；页面简介可能属于其他 edition，seller description 可能含营销文本、跨语言模板或错误 ISBN，所有抽取值必须带 `evidence_id` 和 confidence。


## 推荐的下一步

先以 50–100 个 ISBN 做不落盘的字段覆盖评估：统计每个候选字段的非空率、版本冲突率、图片可访问率和模型抽取置信度。若授权已确认，再实现 SWS XML 解析器、代表 listing 选择器和四张表的 JSONL/Parquet 输出；若授权未确认，先只实现脱离 AbeBooks 的 schema、解析 fixture 和本地模型抽取测试。

## 参考

- 论文：Yin, Han, Yu, “Truth Discovery with Multiple Conflicting Information Providers on the Web”, IEEE TKDE 20(6), 2008；论文中给出的实验规模和人工封面真值见 [PDF](https://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Trust/kdd07_xyin.pdf)。
- [AbeBooks Search Web Services 概览](https://www.abebooks.com/developer/search-web-services/overview)；[参数和返回字段](https://www.abebooks.com/developer/search-web-services/parameters)；[2025 End User Guide](https://www.abebooks.com/docs/AffiliateProgram/WebServices/end-user-guide.pdf)。
- [AbeBooks Terms and Conditions](https://www.abebooks.com/docs/legal/termsandconditions.shtml)；[SWS Terms of Use](https://www.abebooks.com/docs/affiliateprogram/webservices/terms-printable.shtml)；[robots.txt](https://www.abebooks.com/robots.txt)。
