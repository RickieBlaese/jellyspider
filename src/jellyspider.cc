#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <cstdint>

#include <dlfcn.h>
#include <gmp.h>

#include "mpreal/mpreal.h"


#ifndef WORK_SO_FILENAME
#define WORK_SO_FILENAME "libjellyspider-work.so"
#endif

#define STRINGIZE_NX(A) #A

#define STRINGIZE(A) STRINGIZE_NX(A)

#define VERSIONA 0
#define VERSIONB 1
#define VERSIONC 0
#define VERSION STRINGIZE(VERSIONA) "." STRINGIZE(VERSIONB) "." STRINGIZE(VERSIONC)

/* NOLINTBEGIN */
#define DEF_PTRS(X) \
    using sptr ## X ## _t = std::shared_ptr<X ## _t>; \
    using wptr ## X ## _t = std::weak_ptr<X ## _t>;

#define DEF_PTRFUN(X) \
    sptr ## X ## _t make_ ## X () { \
        sptr ## X ## _t e = std::make_shared<X ## _t>(); \
        e->self = e; \
        return e; \
    }
/* NOLINTEND */

#ifdef DEBUG_BUILD

#define ERR_EXIT(A, ...) { \
    std::fprintf(stderr, "jspider: error: file " __FILE__ ":%i in %s(): ", __LINE__, __func__); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
    std::exit(static_cast<int>(A)); \
}

#define WARN(...) { \
    std::fprintf(stderr, "jspider: warning: file " __FILE__ ":%i in %s(): ", __LINE__, __func__); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
}

#else

#define ERR_EXIT(A, ...) { /* NOLINT */ \
    std::fputs("jspider: error: ", stderr); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
    std::exit(static_cast<int>(A)); \
}

#define WARN(...) { /* NOLINT */ \
    std::fputs("jspider: warning: ", stderr); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
}

#endif

template <typename T>
void vec_uniqueify(std::vector<T> &v, const std::function<bool (const T &, const T &)> &eq) {
    for (std::size_t i = 0; i < v.size(); i++) {
        for (std::size_t j = 0; j < v.size(); j++) {
            if (i == j) {
                continue;
            }
            if (eq(v[i], v[j])) {
                v.erase(v.begin() + j);
                j--;
            }
        }
    }
}

/* not thread safe */
std::wstring str_to_wstr(const std::string &s) {
    /* copied and edited from https://stackoverflow.com/a/2573845/19321937 */
    std::wstring ws(s.size(), L' '); /* Overestimate number of code points. */
    std::size_t l = std::mbstowcs(&ws[0], s.c_str(), s.size());
    if (l == static_cast<std::size_t>(-1)) {
        ERR_EXIT(1, "could not convert narrow multibyte string %s to wide string", s.c_str());
    }
    ws.resize(l); /* Shrink to fit. */
    return ws;
}

/* not thread safe */
std::string wstr_to_str(const std::wstring &ws) {
    std::string s;
    std::wctomb(nullptr, 0); /* reset conversion state */
    for (const wchar_t &wc : ws) {
        std::string c(MB_CUR_MAX, '\0');
        const int r = std::wctomb(c.data(), wc);
        if (r == -1) {
            std::string t(reinterpret_cast<const char*>(&wc), sizeof(wc)); /* force, i guess */ /* NOLINT */
            std::erase(t, '\0');
            s += t;
            continue;
            /* if (wind != nullptr) {
                deinit_ncurses();
                wind = nullptr;
            }
            std::wcout << "wide string: " << ws;
            ERR_EXIT(1, "could not convert wide string to narrow multibyte string"); */
        }
        c.resize(r);
        s += c;
    }
    return s;
}

template <typename T>
T get_random_int(const T &a, const T &b) {
    static std::random_device device{};
    static std::mt19937 engine(device());
    return std::uniform_int_distribution(a, b)(engine);
}

template <typename T>
T get_random_int() {
    static std::random_device device{};
    static std::mt19937 engine(device());
    return std::uniform_int_distribution<T>()(engine);
}

template <typename T>
T get_random_real(const T &a, const T &b) {
    static std::random_device device{};
    static std::mt19937 engine(device());
    return std::uniform_real_distribution(a, b)(engine);
}

template <typename T>
T get_random_real() {
    static std::random_device device{};
    static std::mt19937 engine(device());
    return std::uniform_real_distribution<T>()(engine);
}

using dec_t = mpfr::mpreal;

struct mpzw_t {
    mpz_t z;
};

/* NOLINTBEGIN */

gmp_randstate_t randstate;

/* even though we use a geometric distribution for generating mpzw, we need a reasonable upper limit for performance considerations (decide when to use) */
mp_bitcnt_t rand_geo_limit = 128;

#define GEN_DEFAULT_BASE 0.9
#define DIST_CHCK_SUBSET_MAX_SIZE 100

/* NOLINTEND */

/* assumes out is initialized */
void mpzw_geometric_dist(mpzw_t &out, const dec_t &base = GEN_DEFAULT_BASE) {
    dec_t a(0, static_cast<mpfr_prec_t>(rand_geo_limit));
    mpfr_urandomb(a.mpfr_ptr(), randstate); /* now a is in [0, 1) */
    a = mpfr::log(1.0 - a);
    a /= mpfr::log(base);
    mpfr_sqr(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    if (static_cast<bool>(get_random_int<std::uint8_t>(0, 1))) {
        mpfr_neg(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    }
    mpfr_get_z(out.z, a.mpfr_srcptr(), dec_t::get_default_rnd());
}

void dec_compress(dec_t &a, const dec_t &base = GEN_DEFAULT_BASE) {
    mpfr_sqrt(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    mpfr_pow(a.mpfr_ptr(), base.mpfr_srcptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    mpfr_ui_sub(a.mpfr_ptr(), 1, a.mpfr_srcptr(), dec_t::get_default_rnd());
}


double double_compress(double a, const double &base = GEN_DEFAULT_BASE) {
    a = std::sqrt(a);
    a = std::pow(base, a);
    return 1 - a;
}

/* assumes a > 0 */
double mpzw_compress_double(const mpzw_t &a, const dec_t &base = GEN_DEFAULT_BASE) {
    dec_t r = a.z;
    dec_compress(r, base);
    return r.toDouble();
}

struct range_t {
    std::size_t a = 0, b = 0;
};

struct matched_pair_t {
    range_t x, y;
};

template <typename T>
double vec_similarity(const std::vector<T> &x, const std::vector<T> &y) {
    std::vector<matched_pair_t> s;
    for (std::int64_t ai = x.size() - 1; ai >= 0; ai--) {
        for (std::int64_t bi = y.size() - 1; bi >= 0; bi--) {
            if (x[ai] == y[bi]) {
                /* check for overlap */
                bool skip = false;
                for (const matched_pair_t &mp : s) {
                     if ((mp.x.a <= ai && ai < mp.x.b) || (mp.y.a <= bi && bi < mp.y.b)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }

                s.push_back(matched_pair_t{range_t{static_cast<std::size_t>(ai - 1), static_cast<std::size_t>(ai)}, range_t{static_cast<std::size_t>(bi - 1), static_cast<std::size_t>(bi)}});
                for (ai--, bi--; ai >= 0 && bi >= 0; ai--, bi--) {
                    if (x[ai] != y[bi]) {
                        break;
                    }
                }
                s[s.size() - 1].x.a = ai;
                s[s.size() - 1].y.a = bi;
            }
        }
    }

    std::size_t total_len = 0;
    for (const matched_pair_t &mp : s) {
        total_len += mp.x.b - mp.x.a;
    }

    return static_cast<double>(total_len)/static_cast<double>(std::min(x.size(), y.size()));
}

enum struct otype_t : std::uint16_t {
    none,

    /* literal */
    bool_v,
    /* literal */
    int_v,
    /* literal */
    dec_v,

    map_t,

    set_t, /* {...} */
    tuple_t, /* (...) */

    symbol_t, /* \symbol name */

    unresolvable_t /* \unresolvable */
};

/* name has value when it's an unspecified object - i.e. with .v as monostate 
 * note that however, .v being monostate does not imply unspecified - it could be a set objexpr_t with other child exprs, mapcall, or tuple
 * we don't define any constructors / assignment operators here other than the copy assignment operator because objexpr_t
 * handles the rest - in other words, only use obj_t in conjunction with an objexpr_t!! */
struct obj_t { /* NOLINT */
    std::optional<std::wstring> name;
    otype_t type = otype_t::none;
    std::variant<std::monostate, bool, mpzw_t, dec_t> v; /* bool_v, int_v, and dec_v */

    bool operator==(const obj_t &other) const {
        bool res = type == other.type && name.has_value() == other.name.has_value() && v.index() == other.v.index();
        if (v.index() > 0 && v.index() == other.v.index()) {
            std::visit([&res, &other](const auto &val) {
                if constexpr (std::is_same_v<decltype(val), bool>) {
                    res = res && val == std::get<bool>(other.v);
                } else if constexpr (std::is_same_v<decltype(val), mpzw_t>) {
                    res = res && !mpz_cmp(val, std::get<mpzw_t>(other.v));
                } else if constexpr (std::is_same_v<decltype(val), dec_t>) {
                    res = res && val == std::get<dec_t>(other.v);
                }
            }, v);
        }
        if (name.has_value() && other.name.has_value()) {
            res = res && name.value() == other.name.value();
        }
        return res;
    }

    obj_t &operator=(const obj_t &other) { /* assumes v is std::monostate */
        name = other.name;
        type = other.type;
        std::visit([this](const auto &val) {
            if constexpr (std::is_same_v<decltype(val), mpzw_t>) {
                v = mpzw_t{};
                auto &tz = std::get<mpzw_t>(v);
                mpz_init_set(tz, std::get<mpzw_t>(val));
            } else {
                v = val;
            }
        }, other.v);
        return *this;
    }

    bool is_lit_obj() const {
        return (type == otype_t::bool_v || type == otype_t::int_v || type == otype_t::dec_v) && v.index() > 0;
    }
};

template <typename T>
struct bucket_avg_t {
    T s = static_cast<T>(0);
    std::uint64_t c = 0;

    void add(const T &v) {
        s += v;
        c++;
    }

    T avg_or(const T &v) {
        if (c > 0) {
            return s / static_cast<T>(c);
        } else {
            return v;
        }
    }
};

/* each comment specifies how to parse the expression
 * in parentheses after, the guaranteed return otype might be specified,
 * otherwise if not there, it could be any otype
 * as a result of the text representations of andexpr_t and similar expressions,
 * we restrict object names in parsing to not start with a backslash */
enum struct etype_t : std::uint16_t {
    none,

    objexpr_t, /* check otype_t */

    /* number operations */
    addexpr_t, /* a + b + ... */
    negexpr_t, /* - b */
    subexpr_t, /* a - b - ... */

    /* we will not support name smashing like ab to represent a*n, or a b */
    mulexpr_t, /* a * b * ... */

    divexpr_t, /* a / b */
    powexpr_t, /* a ^ b */

    eqexpr_t, /* a = b = ... (-> bool) */
    andexpr_t, /* a \and b \and ... (-> bool) */
    orexpr_t, /* a \or b \or ... (-> bool) */
    notexpr_t, /* \not a (-> bool) */

    containedexpr_t, /* a \in b (-> bool) */
    subsetexpr_t, /* a \subset b \subset ... (-> bool) */

    unionexpr_t, /* a \union b \union ... (-> set) */

    /* change? */
    intersectexpr_t, /* a \inters b \inters ... (-> bool) */

    /* change? ideally we want all of these to be short like < 6 chars. note latex uses \setminus */
    subtractexpr_t, /* a \minus b (-> set) */

    /* a should be an expression which uses referenced objects which are either defined in b or referenced in b
     * b must evaluate to a bool
     * this can produce any a which satisfy b
     */
    stexpr_t, /* a \st b */

    /* a should be an expression which uses referenced objects which are either defined in b or referenced in b
     * b must evaluate to a bool
     * this specifies a set which has elements equal to all possible a that satisfy b
     */
    setspecexpr_t, /* {a | b} */

    /* a is a tuple, n must evaluate to an obj int >= 0 */
    tuplegetexpr_t, /* a[n] */

    /* a and b are sets
     * f is referenced obj map
     * this checks if f is a map from a to b */
    mapspecexpr_t, /* f: a -> b (-> bool) */

    /* a, b are sets
     * f is a referenced obj map */
    mapdefexpr_t, /* \map f: a -> b := x -> c */

    /* we store a weak ptr to the mapdefexpr_t for the function f. recall all names are global.
     * at creation time, try to find the mapdefexpr_t for the function f
     * additionally, this only consumes one more expr. this expr can either be PARSED as
     * 1. a tuple,
     *    in which case each element of the tuple gets passed to f as expected
     * 2. anything else,
     *    in which case the expr gets passed as the first element to f
     * note that mapcalls have the highest precedence in parsing */
    mapcallexpr_t, /* f X */

};

struct expr_t;

DEF_PTRS(expr);

struct thm_t;

using tid_t = std::uint64_t;

std::unordered_map<tid_t, thm_t> thms; /* NOLINT */

tid_t gen_thm_id() {
    static std::uint64_t counter = 0;
    return ++counter;
}

struct thm_t {
    tid_t id;
    std::wstring name;

    sptrexpr_t e; /* we assume e evaluates to a litobjexpr_t with obj holding true */

    explicit thm_t() : id(gen_thm_id()) {}
};

struct pfstep_t {
    sptrexpr_t expr;

    tid_t id; /* id of the theorem used to get to expr */
};

/* std::vector<tid_t> find_thms(sr) {
} */

struct vec_t {
    std::vector<double> v;

    vec_t() = default;
    vec_t(const vec_t &other) = default;
    vec_t(vec_t &&other) = default;
    ~vec_t() = default;

    vec_t &operator=(const vec_t &other) = default;
    vec_t &operator=(vec_t &&other) = default;

    vec_t operator+(const vec_t &other) const {
        std::size_t sz = std::max(v.size(), other.v.size());
        std::size_t minsz = std::min(v.size(), other.v.size());
        vec_t r;
        r.v.resize(sz);
        for (std::size_t i = 0; i < minsz; i++) {
            r.v[i] = v[i] + other.v[i];
        }
        return r;
    }

    void operator+=(const vec_t &other) {
        std::size_t minsz = std::min(v.size(), other.v.size());
        for (std::size_t i = 0; i < minsz; i++) {
            v[i] += other.v[i];
        }
    }

    vec_t operator-(const vec_t &other) const {
        std::size_t sz = std::max(v.size(), other.v.size());
        std::size_t minsz = std::min(v.size(), other.v.size());
        vec_t r;
        r.v.resize(sz);
        for (std::size_t i = 0; i < minsz; i++) {
            r.v[i] = v[i] - other.v[i];
        }
        return r;
    }

    vec_t operator-() const {
        vec_t r = *this;
        for (std::size_t i = 0; i < v.size(); i++) {
            r.v[i] = -r.v[i];
        }
        return r;
    }

    vec_t operator-=(const vec_t &other) const {
        std::size_t sz = std::max(v.size(), other.v.size());
        std::size_t minsz = std::min(v.size(), other.v.size());
        vec_t r;
        r.v.resize(sz);
        for (std::size_t i = 0; i < minsz; i++) {
            r.v[i] = v[i] - other.v[i];
        }
        return r;
    }

    double dot(const vec_t &other) const {
        std::size_t minsz = std::min(v.size(), other.v.size());
        double s = 0;
        for (std::size_t i = 0; i < minsz; i++) {
            s += v[i] * other.v[i];
        }
        return s;
    }

    double metric_euc() const {
        double m = 0;
        for (const double &i : v) {
            m += i * i;
        }
        return std::sqrt(m);
    }

    double distance_euc(const vec_t &other) const {
        std::size_t minsz = std::min(v.size(), other.v.size());
        double m = 0;
        for (std::size_t i = 0; i < minsz; i++) {
            m += (v[i] - other.v[i]) * (v[i] - other.v[i]);
        }
        return std::sqrt(m);
    }
};

/* overall type */
struct oatype_t {
    etype_t e = etype_t::none;
    std::optional<otype_t> o;
};

oatype_t get_oatype(const sptrexpr_t &e);

/* does not move or copy */
template <typename T>
sptrexpr_t to_expr(std::shared_ptr<T> e) {
    return std::dynamic_pointer_cast<expr_t>(e);
}

/* does not move or copy */
template <typename T>
std::shared_ptr<T> from_expr(sptrexpr_t e) {
    return std::dynamic_pointer_cast<T>(e);
}

struct objexpr_t;

DEF_PTRS(objexpr);

/* NOLINTBEGIN */

/* loaded with dlopen on startup
 * can return an empty sptrexpr_t if it wasn't going to change it */
sptrexpr_t (*expr_work)(const expr_t &e, const sptrexpr_t &target, const decltype(thms) &thms);

/* NOLINTEND */

struct expr_t { /* NOLINT */
    wptrexpr_t self;
    std::vector<sptrexpr_t> exprs;
    std::vector<wptrexpr_t> parents;
    /* pair is object, most immediate parent expr */
    std::vector<std::pair<wptrobjexpr_t, wptrexpr_t>> all_objs;
    etype_t type = etype_t::none;
    sptrexpr_t work_cache, calc_cache; /* calc_cache doesn't hold a value when calculate failed */
    bool work_dirty = true, calc_dirty = true;

    explicit expr_t() = default;
    explicit expr_t(decltype(exprs) exprs, decltype(parents) parents) : exprs(std::move(exprs)), parents(std::move(parents)) {}
    virtual ~expr_t() = default;

    void clear() {
        for (sptrexpr_t &expr : exprs) {
            expr->clear();
        }
        exprs.clear();
    }

    void add_obj(const wptrobjexpr_t &o, const sptrexpr_t &parent);

    void add(const sptrexpr_t &expr) {
        if (expr->type == etype_t::objexpr_t) {
            add_obj(std::dynamic_pointer_cast<objexpr_t>(expr), self.lock());
        }
        expr->parents.push_back(self);
        exprs.emplace_back(expr);
    }

    template <typename T>
    void add(const std::shared_ptr<T> &expr) {
        if (expr->type == etype_t::objexpr_t) {
            add_obj(std::dynamic_pointer_cast<objexpr_t>(expr), self.lock());
        }
        expr->parents.push_back(self);
        exprs.emplace_back(to_expr(expr));
    }

    /* does not check any expired parents in expr->parents */
    void del(const sptrexpr_t &expr) {
        std::erase_if(expr->parents, [&](const wptrexpr_t &e) {
            if (!e.expired()) {
                return e.lock() == self.lock();
            }
            return false;
        });
        std::erase(exprs, expr);
    }


    bool is_calc_form() const {
        if (type != etype_t::objexpr_t) { return false; }
        bool calc_form = true;
        for (const sptrexpr_t &expr: exprs) {
            calc_form = calc_form && expr->is_calc_form();
        }
        return calc_form;
    }

    /* walks */
    bool operator==(const expr_t &other) const {
        return false;
    }
    /* {
        bool res = type == other.type;
        res = res && obj.has_value() == other.obj.has_value();
        if (obj.has_value() && other.obj.has_value()) {
            res = res && obj.value() == other.obj.value();
        }
        for (std::uint32_t i = 0; i < exprs.size(); i++) {
            res = res && (*exprs[i] == *other.exprs[i]);
        }
        return res;
    } */


    /* walks (deep copy)
     * does not copy parent
     */
    template <typename T>
    sptrexpr_t base_copy() const {
        sptrexpr_t e = to_expr(std::make_shared<T>());
        e->self = e;
        e->type = type;
        e->exprs.resize(exprs.size());
        for (std::size_t i = 0; i < exprs.size(); i++) {
            e->exprs[i] = exprs[i]->copy();
            e->exprs[i]->parents.push_back(e);
        }
        for (const sptrexpr_t &expr : e->exprs) {
            if (expr->type == etype_t::objexpr_t) {
                e->all_objs.emplace_back(from_expr<objexpr_t>(expr), e->self);
            }
            e->all_objs.insert(e->all_objs.end(), expr->all_objs.begin(), expr->all_objs.end());
        }
        std::function<bool (const std::pair<wptrobjexpr_t, wptrexpr_t> &, const std::pair<wptrobjexpr_t, wptrexpr_t> &)> peq = [](const std::pair<wptrobjexpr_t, wptrexpr_t> &a, const std::pair<wptrobjexpr_t, wptrexpr_t> &b) {
            if (a.first.expired() || a.second.expired() || b.first.expired() || b.second.expired()) {
                return false;
            } else {
                return a.first.lock() == b.first.lock() && a.second.lock() == b.second.lock();
            }
        };
        vec_uniqueify(e->all_objs, peq);
        e->work_dirty = work_dirty;
        e->calc_dirty = calc_dirty;
        if (!work_dirty) {
            e->work_cache = work_cache->copy();
        }
        if (!calc_dirty) {
            e->calc_cache = calc_cache->copy();
        }
        return e;
    }

    virtual sptrexpr_t copy() const {
        return base_copy<expr_t>();
    }

    void walk(const std::function<void (sptrexpr_t &)> &f) {
        for (sptrexpr_t &expr : exprs) {
            f(expr);
            expr->walk(f);
        }
    }

    void walk_const(const std::function<void (const sptrexpr_t &)> &f) const {
        for (const sptrexpr_t &expr : exprs) {
            f(expr);
            expr->walk_const(f);
        }
    }

    void get_deepest_type_stacks(std::vector<std::vector<etype_t>> &out, std::vector<etype_t> &cur) const {
        cur.push_back(type);
        if (exprs.empty()) {
            out.push_back(cur);
            return;
        }
        for (const sptrexpr_t &e : exprs) {
            e->get_deepest_type_stacks(out, cur);
        }
        cur.erase(cur.end() - 1);
    }

    /* walks */
    std::uint32_t size() const {
        std::uint32_t sum = 1;
        walk_const([&sum](const sptrexpr_t &expr) { sum += expr->size() + 1; });
        return sum;
    }

    double dist_randgen(const expr_t &other, std::size_t sample_size) const;

    double dist_match(const expr_t &other) const;

    /* dist are between 0 and 1 */
    double dist(const expr_t &other, std::size_t sample_size) const;

    std::wstring disp_to_str(const std::optional<wptrexpr_t> &parent = {}) const {
        std::wstringstream s;
        disp(s, parent);
        return s.str();
    }

    /* always assumes previous state was ok */
    void signal_dirty() {
        /* if we are fully dirty and previous state was ok, then all parents higher must also be fully dirty */
        if (work_dirty && calc_dirty) {
            return;
        }
        work_dirty = true;
        calc_dirty = true;
        for (wptrexpr_t &parent : parents) {
            if (!parent.expired()) {
                parent.lock()->signal_dirty();
            }
        }
    }

    /* if you want a redone work-ed expr that isn't from work_cache, just call expr_work */
    sptrexpr_t load(bool can_calculate) {
        if (can_calculate) {
            if (calc_dirty) {
                std::optional<sptrexpr_t> o = calculate();
                if (o.has_value()) {
                    calc_cache = std::move(o.value());
                }
                calc_dirty = false;
            }
            if (calc_cache) {
                return calc_cache->copy();
            }
        }
        if (work_dirty) {
            work_cache = expr_work(*this, {}, thms);
            work_dirty = false;
        }
        return work_cache->copy();
    }

    /* if optional has no value, it could not be calculated
     * otherwise should return an objexpr_t, with ALL (recursive) child exprs also objexpr_t - this is the only valid calcuated form (note we don't say "closed form" - it is more subjective and hazy, it depends on the context)
     * it is up to each derived expr_t class to determine precision requirements (DEAL WITH IT!!!)
     */
    virtual std::optional<sptrexpr_t> calculate() const {
        return {};
    }

    virtual sptrexpr_t generate() const {
        return copy();
    }

    /* checks if needs paren */
    void disp_exprs_sep(std::wostream &os, const std::wstring &sep, const decltype(exprs)::const_iterator &begin, const decltype(exprs)::const_iterator &end) const {
        if (exprs.empty()) {
            return;
        }
        for (auto i = begin; i != end - 1; i++) {
            (*i)->disp_check_paren(os, self);
            os << sep;
        }
        (*(end - 1))->disp_check_paren(os, self);
    }

    void disp_check_paren(std::wostream &os, const std::optional<wptrexpr_t> &parent) const {
        bool np = disp_needs_paren(parent);
        if (np) {
            os << L'(';
        }
        disp(os, parent);
        if (np) {
            os << L')';
        }
    }

    virtual bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const {
        return true;
    }

    virtual void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const {
        os << L"_exprtype" << static_cast<std::uint16_t>(type) << L'{';
        disp_exprs_sep(os, L", ", exprs.begin(), exprs.end());
        os << L'}';
    }
};


struct objexpr_t : virtual expr_t {
    obj_t obj;

    sptrexpr_t generate() const override;

    sptrexpr_t copy() const override;

    objexpr_t() {
        type = etype_t::objexpr_t;
    }

    ~objexpr_t() override {
        if (obj.type == otype_t::int_v) {
            mpz_clear(std::get<mpzw_t>(obj.v).z);
        }
    }


    std::optional<sptrexpr_t> calculate() const override {
        return copy();
    }

    /* dist are between 0 and 1 */
    /* assumes obj types match, and this expr is_calc_form */
    double obj_dist(const objexpr_t &other) {
        switch (obj.type) {
            case otype_t::bool_v:
                return static_cast<double>(std::get<bool>(obj.v) != std::get<bool>(other.obj.v));
            case otype_t::int_v: {
                mpzw_t diff{};
                mpz_sub(diff.z, std::get<mpzw_t>(obj.v).z, std::get<mpzw_t>(other.obj.v).z);
                mpz_abs(diff.z, diff.z);
                return mpzw_compress_double(diff);
            }
            case otype_t::dec_v: {
                dec_t a = std::get<dec_t>(obj.v) - std::get<dec_t>(other.obj.v);
                mpfr_abs(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
                dec_compress(a);
                return a.toDouble();
            }

            /* TODO: work on these!! */
            case otype_t::tuple_t:
            case otype_t::symbol_t:
            case otype_t::none:
            case otype_t::unresolvable_t:
                return 1.0;
            case otype_t::set_t: {
                if (exprs.size() > DIST_CHCK_SUBSET_MAX_SIZE || other.exprs.size() > DIST_CHCK_SUBSET_MAX_SIZE) {
                    return 0.01 + 0.99*std::sqrt(double_compress(std::abs(static_cast<double>(exprs.size()) - static_cast<double>(other.exprs.size())))); /* sqrt to push it higher towards being higher dist since they're different length */
                }
                for (const sptrexpr_t &expr : exprs) {
                }
                break;
            }

            default:
                return 1.0;
        }
    }

    bool is_ref_obj() const {
        return obj.name.has_value() && exprs.empty() && obj.v.index() == 0;
    }

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return false;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        switch (obj.type) {
            case otype_t::bool_v:
                os << (std::get<bool>(obj.v) ? L"true" : L"false");
                return;
            case otype_t::int_v: {
                char *s = nullptr;
                gmp_asprintf(&s, "%Zd", std::get<mpzw_t>(obj.v).z);
                std::string rs(s);
                free(s); /* NOLINT */
                os << str_to_wstr(rs);
                return;
            }
            case otype_t::dec_v:
                os << str_to_wstr(std::get<dec_t>(obj.v).toString());
                return;
            case otype_t::map_t:
                os << obj.name.value();
                return;
            case otype_t::set_t:
                os << L'{';
                disp_exprs_sep(os, L", ", exprs.begin(), exprs.end());
                os << L'}';
                return;
            case otype_t::tuple_t:
                os << L'(';
                disp_exprs_sep(os, L", ", exprs.begin(), exprs.end());
                os << L')';
                return;
            case otype_t::symbol_t:
                os << L'\\';
                os << obj.name.value();
                os << L' ';
                return;
            case otype_t::unresolvable_t:
                os << L"_unresolvable";
                return;
            case otype_t::none:
                if (is_ref_obj()) {
                    os << obj.name.value();
                } else {
                    os << L"_noneobj";
                }
                return;
            default:
                os << L"_objexprtype" << static_cast<std::uint16_t>(obj.type) << '{';
                disp_exprs_sep(os, L", ", exprs.begin(), exprs.end());
                os << '}';
                return;
        }
    }
};

DEF_PTRFUN(objexpr);


/* does not touch .parents or .exprs */
void expr_t::add_obj(const wptrobjexpr_t &o, const sptrexpr_t &parent) {
    all_objs.emplace_back(o, parent);
    for (wptrexpr_t &p : parents) {
        if (!p.expired()) {
            p.lock()->add_obj(o, parent);
        }
    }
}

oatype_t get_oatype(const sptrexpr_t &e) {
    if (e->type == etype_t::objexpr_t) {
        return oatype_t{.e = e->type, .o = dynamic_cast<objexpr_t*>(e.get())->obj.type};
    }
    return oatype_t{.e = e->type};
}

sptrexpr_t objexpr_t::copy() const {
    sptrobjexpr_t oe = from_expr<objexpr_t>(base_copy<objexpr_t>());
    oe->obj = obj;
    if (obj.type == otype_t::int_v) {
        oe->obj.v = mpzw_t{};
        auto &nz = std::get<mpzw_t>(oe->obj.v);
        mpz_init_set(nz.z, std::get<mpzw_t>(obj.v).z);
    }
    return to_expr(oe);
}

sptrobjexpr_t make_bool(bool v) {
    sptrobjexpr_t oe = make_objexpr();
    oe->obj.type = otype_t::bool_v;
    oe->obj.v = v;
    return oe;
}

/* uses mpz_init_set to copy */
sptrobjexpr_t make_mpzw_copy(mpzw_t i) {
    sptrobjexpr_t oe = make_objexpr();
    oe->obj.type = otype_t::int_v;
    oe->obj.v = mpzw_t{};
    auto &tz = std::get<mpzw_t>(oe->obj.v);
    mpz_init_set(tz.z, i.z);
    return oe;
}

/* steals mpz */
sptrobjexpr_t make_mpzw(mpzw_t i) {
    sptrobjexpr_t oe = make_objexpr();
    oe->obj.type = otype_t::int_v;
    oe->obj.v = mpzw_t{};
    std::get<mpzw_t>(oe->obj.v) = i;
    return oe;
}

sptrobjexpr_t make_dec(const dec_t &a) {
    sptrobjexpr_t oe = make_objexpr();
    oe->obj.type = otype_t::dec_v;
    oe->obj.v = a;
    return oe;
}

/* this is hard... for non lit objs */
sptrexpr_t objexpr_t::generate() const {
    switch (obj.type) {
        case otype_t::bool_v:
            return make_bool(static_cast<bool>(get_random_int<std::uint8_t>(0, 1)));
        case otype_t::int_v: {
            mpzw_t a{};
            mpz_init(a.z);
            mpzw_geometric_dist(a);
            return make_mpzw_copy(a);
        }
        case otype_t::dec_v: {
            dec_t a;
            mpfr_urandom(a.mpfr_ptr(), randstate, dec_t::get_default_rnd());
            return make_dec(a);
        }
        case otype_t::unresolvable_t:
        case otype_t::symbol_t:
        case otype_t::none:

        /* TODO: work on these !! hard :( */
        case otype_t::set_t:
        case otype_t::map_t:
        case otype_t::tuple_t:
            return copy();
    }
}

struct mapdefexpr_t;

DEF_PTRS(mapdefexpr);

std::unordered_map<std::wstring, wptrmapdefexpr_t> funcdefs; /* NOLINT */

struct mapdefexpr_t : virtual expr_t {
    std::wstring func_name;
    std::vector<std::wstring> param_names;

    sptrexpr_t func_body() {
        return exprs[2];
    }

    mapdefexpr_t() {
        type = etype_t::mapdefexpr_t;
    }

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L"\\map ";
        os << func_name;
        os << L": ";
        exprs[0]->disp(os, self);
        os << L" -> ";
        exprs[1]->disp(os, self);
        os << L" := ";
        if (param_names.size() > 1) {
            os << L'(';
        }
        for (std::size_t i = 0; i < param_names.size() - 1; i++) {
            os << param_names[i] << L", ";
        }
        if (!param_names.empty()) {
            os << param_names[param_names.size() - 1];
        }
        if (param_names.size() > 1) {
            os << L')';
        }
        os << L" -> ";
        exprs[2]->disp(os, self);
    }
};

/* assumes funcdefs does not already have md.lock()->func_name */
void expr_add_to_defs(const wptrmapdefexpr_t &md) {
    funcdefs[md.lock()->func_name] = md;
}

DEF_PTRFUN(mapdefexpr);

struct mapcallexpr_t : virtual expr_t {

    mapcallexpr_t() {
        type = etype_t::mapcallexpr_t;
    }

    std::optional<sptrexpr_t> calculate() const override {
        std::optional<sptrexpr_t> ofnres = exprs[0]->calculate();
        if (!ofnres.has_value()) {
            return {};
        }
        sptrexpr_t &fnres = ofnres.value();
        if (fnres->type != etype_t::objexpr_t) {
            ERR_EXIT(1, "mapcallexpr_t: referenced function did not calculate to an objexpr_t, was %s", wstr_to_str(fnres->disp_to_str()).c_str());
        }
        sptrobjexpr_t fn = from_expr<objexpr_t>(std::move(fnres));
        if (!fn->obj.name.has_value()) {
            ERR_EXIT(1, "mapcallexpr_t: referenced function calculated to an anonymous objexpr_t, was %s", wstr_to_str(fnres->disp_to_str()).c_str());
        }
        if (!funcdefs.contains(fn->obj.name.value())) {
            ERR_EXIT(1, "mapcallexpr_t: referenced function calculated to a function name which was not defined previously, was %s", wstr_to_str(fn->obj.name.value()).c_str());
        }
        wptrmapdefexpr_t mapdef = funcdefs[fn->obj.name.value()];
        if (mapdef.expired()) {
            ERR_EXIT(1, "mapcallexpr_t: referenced function calculated to function name %s which was in funcdefs but with expired wptrmapdefexpr_t", wstr_to_str(fn->obj.name.value()).c_str());
        }
        sptrexpr_t fnbody = mapdef.lock()->func_body()->copy();
        if (mapdef.lock()->param_names.size() != exprs.size() - 1) {
            ERR_EXIT(1, "mapcallexpr_t: incorrect number of parameters passed to function name %s, expected %zu, got %zu", wstr_to_str(fn->obj.name.value()).c_str(), mapdef.lock()->param_names.size(), exprs.size() - 1);
        }
        for (std::size_t pi = 0; pi < mapdef.lock()->param_names.size(); pi++) {
            const std::wstring &param_name = mapdef.lock()->param_names[pi];
            for (const auto &[obj, parent] : fnbody->all_objs) {
                if (obj.expired()) {
                    continue;
                }
                if (obj.lock()->obj.name.has_value()) {
                    if (obj.lock()->obj.name.value() == param_name) {
                        sptrexpr_t sparent = parent.lock();
                        std::replace(sparent->exprs.begin(), sparent->exprs.end(), to_expr(obj.lock()), exprs[pi + 1]);
                        sparent->signal_dirty();
                    }
                }
            }
        }
        return fnbody->calculate();
    }

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L"\\call ";
        exprs[0]->disp(os, self);
        os << L'(';
        disp_exprs_sep(os, L", ", exprs.begin() + 1, exprs.end());
        os << L')';
    }
};

DEF_PTRS(mapcallexpr);
DEF_PTRFUN(mapcallexpr);

struct addexpr_t : virtual expr_t {
    addexpr_t() {
        type = etype_t::addexpr_t;
    }

    std::optional<sptrexpr_t> calculate() const override {
        std::variant<mpzw_t, dec_t> sum = mpzw_t{};
        mpz_init(std::get<mpzw_t>(sum).z);
        for (const sptrexpr_t &expr : exprs) {
            std::optional<sptrexpr_t> res = expr->calculate();
            if (!res.has_value()) {
                return {};
            }
            if (res.value()->type != etype_t::objexpr_t) {
                return {}; /* TODO: should this error? */
            }
            sptrobjexpr_t ores = from_expr<objexpr_t>(std::move(res.value()));
            if (sum.index() == 0) { /* is mpz */
                if (ores->obj.type == otype_t::dec_v) {
                    dec_t tmp = dec_t(std::get<mpzw_t>(sum).z);
                    mpz_clear(std::get<mpzw_t>(sum).z);
                    sum = tmp + std::get<dec_t>(ores->obj.v);
                } else if (ores->obj.type == otype_t::int_v) {
                    mpz_add(std::get<mpzw_t>(sum).z, std::get<mpzw_t>(sum).z, std::get<mpzw_t>(ores->obj.v).z);
                } else {
                    return {}; /* TODO: should this error? */
                }
            } else if (sum.index() == 1) { /* is dec */
                if (ores->obj.type == otype_t::dec_v) {
                    std::get<dec_t>(sum) += std::get<dec_t>(ores->obj.v);
                } else if (ores->obj.type == otype_t::int_v) {
                    std::get<dec_t>(sum) += std::get<mpzw_t>(ores->obj.v).z;
                } else {
                    return {}; /* TODO: should this error? */
                }
            }
        }
        sptrobjexpr_t r = make_objexpr();
        if (sum.index() == 0) { /* is mpz */
            r->obj.v = std::get<mpzw_t>(sum);
            r->obj.type = otype_t::int_v;
        } else if (sum.index() == 1) { /* is dec_t */
            r->obj.v = std::get<dec_t>(sum);
            r->obj.type = otype_t::dec_v;
        }
        return r;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" + ", exprs.begin(), exprs.end());
    }
};

DEF_PTRS(addexpr);
DEF_PTRFUN(addexpr);

sptrexpr_t addexpr_t::copy() const {
    return base_copy<addexpr_t>();
}

struct negexpr_t : virtual expr_t {
    negexpr_t() {
        type = etype_t::negexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L'-';
        exprs[0]->disp_check_paren(os, parent);
    }

    std::optional<sptrexpr_t> calculate() const override {
        std::optional<sptrexpr_t> ores = exprs[0]->calculate();
        if (!ores.has_value()) {
            return {};
        }
        sptrexpr_t res = ores.value();
        if (res->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t oe = from_expr<objexpr_t>(res);
        if (oe->obj.type == otype_t::int_v) {
            auto &i = std::get<mpzw_t>(oe->obj.v);
            mpz_neg(i.z, i.z);
        } else if (oe->obj.type == otype_t::dec_v) {
            auto &r = std::get<dec_t>(oe->obj.v);
            mpfr_neg(r.mpfr_ptr(), r.mpfr_srcptr(), dec_t::get_default_rnd());
        } else {
            return {}; /* TODO: should this error? */
        }
        return oe;
    }
};

sptrexpr_t negexpr_t::copy() const {
    return base_copy<negexpr_t>();
}

DEF_PTRS(negexpr);
DEF_PTRFUN(negexpr);


struct subexpr_t : virtual expr_t {
    subexpr_t() {
        type = etype_t::subexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" - ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        std::variant<mpzw_t, dec_t> sum = mpzw_t{};
        mpz_init(std::get<mpzw_t>(sum).z);
        if (exprs.empty()) {
            return make_mpzw(std::get<mpzw_t>(sum)); /* the mpzw should be 0 here, since we just init-ed */
        }
        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        if (!res.has_value()) {
            return {};
        }
        if (res.value()->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(std::move(res.value()));
        if (ores->obj.type == otype_t::dec_v) {
            mpz_clear(std::get<mpzw_t>(sum).z);
            sum = std::get<dec_t>(ores->obj.v);
        } else if (ores->obj.type == otype_t::int_v) {
            mpz_set(std::get<mpzw_t>(sum).z, std::get<mpzw_t>(ores->obj.v).z);
        }
        for (std::size_t i = 1; i < exprs.size(); i++) {
            const sptrexpr_t &expr = exprs[i];
            std::optional<sptrexpr_t> res = expr->calculate();
            if (!res.has_value()) {
                return {};
            }
            if (res.value()->type != etype_t::objexpr_t) {
                return {}; /* TODO: should this error? */
            }
            sptrobjexpr_t ores = from_expr<objexpr_t>(std::move(res.value()));
            if (sum.index() == 0) { /* is mpz */
                if (ores->obj.type == otype_t::dec_v) {
                    dec_t tmp = dec_t(std::get<mpzw_t>(sum).z);
                    mpz_clear(std::get<mpzw_t>(sum).z);
                    sum = tmp - std::get<dec_t>(ores->obj.v);
                } else if (ores->obj.type == otype_t::int_v) {
                    mpz_sub(std::get<mpzw_t>(sum).z, std::get<mpzw_t>(sum).z, std::get<mpzw_t>(ores->obj.v).z);
                } else {
                    return {}; /* TODO: should this error? */
                }
            } else if (sum.index() == 1) { /* is dec */
                if (ores->obj.type == otype_t::dec_v) {
                    std::get<dec_t>(sum) -= std::get<dec_t>(ores->obj.v);
                } else if (ores->obj.type == otype_t::int_v) {
                    std::get<dec_t>(sum) -= std::get<mpzw_t>(ores->obj.v).z;
                } else {
                    return {}; /* TODO: should this error? */
                }
            }
        }
        sptrobjexpr_t r = make_objexpr();
        if (sum.index() == 0) { /* is mpz */
            r->obj.v = std::get<mpzw_t>(sum);
            r->obj.type = otype_t::int_v;
        } else if (sum.index() == 1) { /* is dec_t */
            r->obj.v = std::get<dec_t>(sum);
            r->obj.type = otype_t::dec_v;
        }
        return r;
    }
};

sptrexpr_t subexpr_t::copy() const {
    return base_copy<subexpr_t>();
}

DEF_PTRS(subexpr);
DEF_PTRFUN(subexpr);


struct mulexpr_t : virtual expr_t {
    mulexpr_t() {
        type = etype_t::mulexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" * ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        std::variant<mpzw_t, dec_t> prod = mpzw_t{};
        mpz_init_set_ui(std::get<mpzw_t>(prod).z, 1);
        for (const sptrexpr_t &expr : exprs) {
            std::optional<sptrexpr_t> res = expr->calculate();
            if (!res.has_value()) {
                return {};
            }
            if (res.value()->type != etype_t::objexpr_t) {
                return {}; /* TODO: should this error? */
            }
            sptrobjexpr_t ores = from_expr<objexpr_t>(std::move(res.value()));
            if (prod.index() == 0) { /* is mpz */
                if (ores->obj.type == otype_t::dec_v) {
                    dec_t tmp = dec_t(std::get<mpzw_t>(prod).z);
                    mpz_clear(std::get<mpzw_t>(prod).z);
                    prod = tmp * std::get<dec_t>(ores->obj.v);
                } else if (ores->obj.type == otype_t::int_v) {
                    mpz_mul(std::get<mpzw_t>(prod).z, std::get<mpzw_t>(prod).z, std::get<mpzw_t>(ores->obj.v).z);
                } else {
                    return {}; /* TODO: should this error? */
                }
            } else if (prod.index() == 1) { /* is dec */
                if (ores->obj.type == otype_t::dec_v) {
                    std::get<dec_t>(prod) *= std::get<dec_t>(ores->obj.v);
                } else if (ores->obj.type == otype_t::int_v) {
                    std::get<dec_t>(prod) *= std::get<mpzw_t>(ores->obj.v).z;
                } else {
                    return {}; /* TODO: should this error? */
                }
            }
        }
        sptrobjexpr_t r = make_objexpr();
        if (prod.index() == 0) { /* is mpz */
            r->obj.v = std::get<mpzw_t>(prod);
            r->obj.type = otype_t::int_v;
        } else if (prod.index() == 1) { /* is dec_t */
            r->obj.v = std::get<dec_t>(prod);
            r->obj.type = otype_t::dec_v;
        }
        return r;
    }
};

sptrexpr_t mulexpr_t::copy() const {
    return base_copy<mulexpr_t>();
}

DEF_PTRS(mulexpr);
DEF_PTRFUN(mulexpr);


struct divexpr_t : virtual expr_t {
    divexpr_t() {
        type = etype_t::divexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" / ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.size() > 2) { /* divexprs like this are ambiguous and we won't allow them */
            ERR_EXIT(1, "divexpr_t: had too many sub expressions, expected at most 2, had %zu", exprs.size());
        }
        std::variant<mpzw_t, dec_t> prod = mpzw_t{};
        mpz_init_set_ui(std::get<mpzw_t>(prod).z, 1);
        if (exprs.empty()) {
            return make_mpzw(std::get<mpzw_t>(prod)); /* the mpzw should be 1 here, since we just init and set */
        }
        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        if (!res.has_value()) {
            return {};
        }
        if (res.value()->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(std::move(res.value()));
        if (ores->obj.type == otype_t::dec_v) {
            mpz_clear(std::get<mpzw_t>(prod).z);
            prod = std::get<dec_t>(ores->obj.v);
        } else if (ores->obj.type == otype_t::int_v) {
            mpz_set(std::get<mpzw_t>(prod).z, std::get<mpzw_t>(ores->obj.v).z);
        }

        std::optional<sptrexpr_t> bres = exprs[1]->calculate();
        if (!bres.has_value()) {
            return {};
        }
        if (bres.value()->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t bores = from_expr<objexpr_t>(std::move(bres.value()));
        if (prod.index() == 0) { /* is mpz */
            if (bores->obj.type == otype_t::dec_v) {
                dec_t tmp = dec_t(std::get<mpzw_t>(prod).z);
                mpz_clear(std::get<mpzw_t>(prod).z);
                prod = tmp / std::get<dec_t>(bores->obj.v);
            } else if (bores->obj.type == otype_t::int_v) {
                dec_t tmp = dec_t(std::get<mpzw_t>(prod).z);
                mpz_clear(std::get<mpzw_t>(prod).z);
                prod = tmp / std::get<mpzw_t>(bores->obj.v).z;
            } else {
                return {}; /* TODO: should this error? */
            }
        } else if (prod.index() == 1) { /* is dec */
            if (bores->obj.type == otype_t::dec_v) {
                std::get<dec_t>(prod) *= std::get<dec_t>(bores->obj.v);
            } else if (bores->obj.type == otype_t::int_v) {
                std::get<dec_t>(prod) *= std::get<mpzw_t>(bores->obj.v).z;
            } else {
                return {}; /* TODO: should this error? */
            }
        }
        sptrobjexpr_t r = make_objexpr();
        if (prod.index() == 0) { /* is mpz. this should be impossible but i'll leave the case here */
            r->obj.v = std::get<mpzw_t>(prod);
            r->obj.type = otype_t::int_v;
        } else if (prod.index() == 1) { /* is dec_t */
            r->obj.v = std::get<dec_t>(prod);
            r->obj.type = otype_t::dec_v;
        }
        return r;
    }
};

sptrexpr_t divexpr_t::copy() const {
    return base_copy<divexpr_t>();
}

DEF_PTRS(divexpr);
DEF_PTRFUN(divexpr);

struct powexpr_t : virtual expr_t {
    powexpr_t() {
        type = etype_t::powexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" ^ ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.size() > 2) { /* powexprs like this are ambiguous and we won't allow them */
            ERR_EXIT(1, "powexpr_t: had too many sub expressions, expected at most 2, had %zu", exprs.size());
        }
        if (exprs.empty()) { /* we also won't allow this */
            ERR_EXIT(1, "powexpr_t: had no sub expressions, expected at least 1");
        }
        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        if (!res.has_value()) {
            return {};
        }
        if (res.value()->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
        dec_t a;
        if (ores->obj.type == otype_t::int_v) {
            a = std::get<mpzw_t>(ores->obj.v).z;
        } else if (ores->obj.type == otype_t::dec_v) {
            a = std::get<dec_t>(ores->obj.v);
        }
        if (exprs.size() == 1) {
            return make_dec(a);
        }
        std::optional<sptrexpr_t> bres = exprs[1]->calculate();
        if (!bres.has_value()) {
            return {};
        }
        if (bres.value()->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t bores = from_expr<objexpr_t>(bres.value());
        dec_t b;
        if (bores->obj.type == otype_t::int_v) {
            b = std::get<mpzw_t>(bores->obj.v).z;
        } else if (bores->obj.type == otype_t::dec_v) {
            b = std::get<dec_t>(bores->obj.v);
        }
        mpfr_pow(a.mpfr_ptr(), a.mpfr_srcptr(), b.mpfr_srcptr(), dec_t::get_default_rnd());
        return make_dec(a);
    }
};

sptrexpr_t powexpr_t::copy() const {
    return base_copy<powexpr_t>();
}

DEF_PTRS(powexpr);
DEF_PTRFUN(powexpr);

struct eqexpr_t : virtual expr_t {
    eqexpr_t() {
        type = etype_t::eqexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" = ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.empty()) {
            ERR_EXIT(1, "eqexpr_t: had no sub expressions, expected at least 1");
        }
        sptrexpr_t orig = exprs[0];
        for (std::size_t i = 1; i < exprs.size(); i++) {
            const sptrexpr_t &expr = exprs[i];
            std::optional<sptrexpr_t> res = expr->calculate();
            if (!res.has_value()) {
                return {};
            }
            if (*res.value() == *orig) { /* TODO: WORK ON!!! */
                return make_bool(false); /* short circuit */
            }
        }
        return make_bool(true);
    }
};

sptrexpr_t eqexpr_t::copy() const {
    return base_copy<eqexpr_t>();
}

DEF_PTRS(eqexpr);
DEF_PTRFUN(eqexpr);


struct andexpr_t : virtual expr_t {
    andexpr_t() {
        type = etype_t::andexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" \\and ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.empty()) {
            ERR_EXIT(1, "andexpr_t: had no sub expressions, expected at least 1");
        }
        for (const sptrexpr_t &expr : exprs) {
            std::optional<sptrexpr_t> res = expr->calculate();
            if (!res.has_value()) {
                return {};
            }
            if (res.value()->type != etype_t::objexpr_t) {
                return {}; /* TODO: should this error? */
            }
            sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
            if (ores->obj.type != otype_t::bool_v) {
                return {}; /* TODO: should this error? */
            }
            if (!std::get<bool>(ores->obj.v)) {
                return make_bool(false); /* short circuit */
            }
        }
        return make_bool(true);
    }
};

sptrexpr_t andexpr_t::copy() const {
    return base_copy<andexpr_t>();
}

DEF_PTRS(andexpr);
DEF_PTRFUN(andexpr);


struct orexpr_t : virtual expr_t {
    orexpr_t() {
        type = etype_t::orexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" \\or ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.empty()) {
            ERR_EXIT(1, "orexpr_t: had no sub expressions, expected at least 1");
        }
        for (const sptrexpr_t &expr : exprs) {
            std::optional<sptrexpr_t> res = expr->calculate();
            if (!res.has_value()) {
                return {};
            }
            if (res.value()->type != etype_t::objexpr_t) {
                return {}; /* TODO: should this error? */
            }
            sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
            if (ores->obj.type != otype_t::bool_v) {
                return {}; /* TODO: should this error? */
            }
            if (std::get<bool>(ores->obj.v)) {
                return make_bool(true); /* short circuit */
            }
        }
        return make_bool(false);
    }
};

sptrexpr_t orexpr_t::copy() const {
    return base_copy<orexpr_t>();
}

DEF_PTRS(orexpr);
DEF_PTRFUN(orexpr);


struct notexpr_t : virtual expr_t {
    notexpr_t() {
        type = etype_t::notexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return !exprs.empty();
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L"\\not ";
        exprs[0]->disp_check_paren(os, parent);
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.empty()) {
            ERR_EXIT(1, "notexpr_t: had no sub expressions, expected exactly 1");
        }
        if (exprs.size() > 1) {
            ERR_EXIT(1, "notexpr_t: had too many sub expressions, expected exactly 1, got %zu", exprs.size());
        }
        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        if (!res.has_value()) {
            return {};
        }
        if (res.value()->type != etype_t::objexpr_t) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
        if (ores->obj.type != otype_t::bool_v) {
            return {}; /* TODO: should this error? */
        }
        std::get<bool>(ores->obj.v) = !std::get<bool>(ores->obj.v);
        return to_expr(ores);
    }
};

sptrexpr_t notexpr_t::copy() const {
    return base_copy<notexpr_t>();
}

DEF_PTRS(notexpr);
DEF_PTRFUN(notexpr);

struct containedexpr_t : virtual expr_t {
    containedexpr_t() {
        type = etype_t::containedexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return !exprs.empty();
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        exprs[0]->disp_check_paren(os, parent);
        os << L" \\in ";
        exprs[1]->disp_check_paren(os, parent);
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.size() < 2) {
            ERR_EXIT(1, "containedexpr_t: had too little sub expressions, expected exactly 2, got %zu", exprs.size());
        }
        if (exprs.size() > 2) {
            ERR_EXIT(1, "containedexpr_t: had too many sub expressions, expected exactly 2, got %zu", exprs.size());
        }
        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        std::optional<sptrexpr_t> setres = exprs[1]->calculate();
        if (!res.has_value() || !setres.has_value()) {
            return {};
        }
        if ((res.value()->type != etype_t::objexpr_t) || (setres.value()->type != etype_t::objexpr_t)) { /* TODO: for now we require all to be obj sets here, but allow setgens in the future */
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
        sptrobjexpr_t osetres = from_expr<objexpr_t>(setres.value());
        if (osetres->obj.type != otype_t::set_t) {
            return {}; /* TODO: should this error? */
        }
        for (const sptrexpr_t &expr : osetres->exprs) { /* note expr should be an objexpr by calculate */
            if (*from_expr<objexpr_t>(expr) == *ores) { /* TODO: WORK ON!!! */
                return make_bool(true);
            }
        }
        return make_bool(false);
    }
};

sptrexpr_t containedexpr_t::copy() const {
    return base_copy<containedexpr_t>();
}

DEF_PTRS(containedexpr);
DEF_PTRFUN(containedexpr);




struct tuplegetexpr_t : virtual expr_t {
    tuplegetexpr_t() {
        type = etype_t::tuplegetexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return !exprs.empty();
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        exprs[0]->disp_check_paren(os, self);
        os << L'[';
        exprs[1]->disp_check_paren(os, self);
        os << L']';
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.size() < 2) {
            ERR_EXIT(1, "tuplegetexpr_t: had too little sub expressions, expected exactly 2, got %zu", exprs.size());
        }
        if (exprs.size() > 2) {
            ERR_EXIT(1, "tuplegetexpr_t: had too many sub expressions, expected exactly 2, got %zu", exprs.size());
        }
        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        std::optional<sptrexpr_t> indres = exprs[1]->calculate();
        if (!res.has_value() || !indres.has_value()) {
            return {};
        }
        if ((res.value()->type != etype_t::objexpr_t) || (indres.value()->type != etype_t::objexpr_t)) {
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
        sptrobjexpr_t oindres = from_expr<objexpr_t>(indres.value());
        if (ores->obj.type != otype_t::tuple_t && oindres->obj.type != otype_t::int_v) {
            return {}; /* TODO: should this error? */
        }
        auto &ind = std::get<mpzw_t>(oindres->obj.v);
        if (mpz_cmp_ui(ind.z, ores->exprs.size()) >= 0) {
            ERR_EXIT(1, "tuplegetexpr_t: expected index %s to be less than the tuple size %zu", wstr_to_str(oindres->disp_to_str()).c_str(), ores->exprs.size());
        }
        if (mpz_cmp_ui(ind.z, 0) < 0) {
            ERR_EXIT(1, "tuplegetexpr_t: expected index %s to be greater than 0", wstr_to_str(oindres->disp_to_str()).c_str());
        }
        return ores->exprs[mpz_get_ui(ind.z)];
    }
};

sptrexpr_t tuplegetexpr_t::copy() const {
    return base_copy<tuplegetexpr_t>();
}

DEF_PTRS(tuplegetexpr);
DEF_PTRFUN(tuplegetexpr);

struct subsetexpr_t : virtual expr_t {
    subsetexpr_t() {
        type = etype_t::subsetexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return exprs.size() > 1;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        disp_exprs_sep(os, L" \\subset ", exprs.begin(), exprs.end());
    }

    std::optional<sptrexpr_t> calculate() const override {
        return {}; /* TODO */
    }
};

sptrexpr_t subsetexpr_t::copy() const {
    return base_copy<subsetexpr_t>();
}

DEF_PTRS(subsetexpr);
DEF_PTRFUN(subsetexpr);


struct mapspecexpr_t : virtual expr_t {
    mapspecexpr_t() {
        type = etype_t::mapspecexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        exprs[0]->disp_check_paren(os, self);
        os << L": ";
        exprs[1]->disp_check_paren(os, self);
        os << L" -> ";
        exprs[2]->disp_check_paren(os, self);
    }

    std::optional<sptrexpr_t> calculate() const override {
        if (exprs.size() != 3) {
            ERR_EXIT(1, "mapspecexpr_t: expected 3 sub expressions, got %zu", exprs.size());
        }

        std::optional<sptrexpr_t> res = exprs[0]->calculate();
        std::optional<sptrexpr_t> ares = exprs[1]->calculate();
        std::optional<sptrexpr_t> bres = exprs[2]->calculate();
        if (!res.has_value() || !ares.has_value() || !bres.has_value()) {
            return {};
        }
        if ((res.value()->type != etype_t::objexpr_t) || (ares.value()->type != etype_t::objexpr_t) || (bres.value()->type != etype_t::objexpr_t)) { /* TODO: for now we require all to be obj sets here, but allow setgens in the future? */
            return {}; /* TODO: should this error? */
        }
        sptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
        sptrobjexpr_t oares = from_expr<objexpr_t>(ares.value());
        sptrobjexpr_t obres = from_expr<objexpr_t>(bres.value());
        if ((ores->obj.type != otype_t::map_t) || (oares->obj.type != otype_t::set_t) || (obres->obj.type != otype_t::set_t)) {
            return {}; /* TODO: should this error? */
        }
        if (!funcdefs.contains(ores->obj.name.value())) { /* TODO: what about anonymous maps? */
            return {}; /* TODO: should this error? */
        }
        wptrmapdefexpr_t mapdef = funcdefs[ores->obj.name.value()];
        std::optional<sptrexpr_t> amdset = mapdef.lock()->exprs[0]->calculate();
        std::optional<sptrexpr_t> bmdset = mapdef.lock()->exprs[1]->calculate();
        if (!amdset.has_value() || !bmdset.has_value()) {
            return {};
        }
        if ((amdset.value()->type != etype_t::objexpr_t) || (bmdset.value()->type != etype_t::objexpr_t)) {
        }
        sptrobjexpr_t oamdset = from_expr<objexpr_t>(amdset.value());
        sptrobjexpr_t obmdset = from_expr<objexpr_t>(bmdset.value());
        if ((oamdset->obj.type != otype_t::set_t) || (obmdset->obj.type != otype_t::set_t)) {
            return {}; /* TODO: should this error? */
        }
        sptrsubsetexpr_t asubset = make_subsetexpr();
        asubset->add(oares);
        asubset->add(oamdset);
        std::optional<sptrexpr_t> asubsetres = asubset->calculate();
        if (!asubsetres.has_value()) {
            return {};
        }
        sptrobjexpr_t aissub = from_expr<objexpr_t>(asubsetres.value());
        if (!std::get<bool>(aissub->obj.v)) {
            return make_bool(false);
        }
        sptrsubsetexpr_t bsubset = make_subsetexpr();
        bsubset->add(obmdset);
        bsubset->add(obres);
        std::optional<sptrexpr_t> bsubsetres = bsubset->calculate();
        if (!bsubsetres.has_value()) {
            return {};
        }
        sptrobjexpr_t bissub = from_expr<objexpr_t>(bsubsetres.value());
        if (!std::get<bool>(bissub->obj.v)) {
            return make_bool(false);
        }
        return make_bool(true);
    }
};

sptrexpr_t mapspecexpr_t::copy() const {
    return base_copy<mapspecexpr_t>();
}

DEF_PTRS(mapspecexpr);
DEF_PTRFUN(mapspecexpr);


struct setspecexpr_t : virtual expr_t {
    setspecexpr_t() {
        type = etype_t::setspecexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L'{';
        exprs[0]->disp_check_paren(os, self);
        os << L" | ";
        exprs[1]->disp_check_paren(os, self);
        os << L'}';
    }

    std::optional<sptrexpr_t> calculate() const override {
        return {};
    }
};


sptrexpr_t setspecexpr_t::copy() const {
    return base_copy<setspecexpr_t>();
}

DEF_PTRS(setspecexpr);
DEF_PTRFUN(setspecexpr);


struct stexpr_t : virtual expr_t {
    stexpr_t() {
        type = etype_t::stexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L'{';
        exprs[0]->disp_check_paren(os, self);
        os << L" \\st ";
        exprs[1]->disp_check_paren(os, self);
        os << L'}';
    }

    std::optional<sptrexpr_t> calculate() const override {
        return {};
    }
};


sptrexpr_t stexpr_t::copy() const {
    return base_copy<stexpr_t>();
}

DEF_PTRS(stexpr);
DEF_PTRFUN(stexpr);


struct unionexpr_t : virtual expr_t {
    unionexpr_t() {
        type = etype_t::unionexpr_t;
    }

    sptrexpr_t copy() const override;

    bool disp_needs_paren(const std::optional<wptrexpr_t> &parent) const override {
        return true;
    }

    void disp(std::wostream &os, const std::optional<wptrexpr_t> &parent) const override {
        os << L'{';
        exprs[0]->disp_check_paren(os, self);
        os << L" \\st ";
        exprs[1]->disp_check_paren(os, self);
        os << L'}';
    }

    std::optional<sptrexpr_t> calculate() const override {
        return {};
    }
};

sptrexpr_t unionexpr_t::copy() const {
    return base_copy<unionexpr_t>();
}


static constexpr std::size_t randgen_default_sample_size = 10;

double expr_t::dist_randgen(const expr_t &other, std::size_t sample_size = randgen_default_sample_size) const {
    sptrexpr_t e = copy(), o = other.copy();
    /* first we collect all objs */
    std::vector<sptrobjexpr_t> all_objs_e;
    /* TODO: !!!!! fix these pointer casts */
    /* e->walk_const([&all_objs_e](const sptrexpr_t &texpr) {
        if (texpr->type == etype_t::objexpr_t && texpr->exprs.empty()) {
            sptrobjexpr_t tobjexpr = from_expr_copy<objexpr_t>(texpr);
            if (tobjexpr->obj.name.has_value()) {
                all_objs_e.push_back(from_expr<objexpr_t>(texpr));
            }
        }
    }); */
    std::vector<sptrobjexpr_t> all_objs_o;
    /* TODO: !!!!! fix these pointer casts */
    /* o->walk_const([&all_objs_o](const sptrexpr_t &texpr) {
        if (texpr->type == etype_t::objexpr_t && texpr->exprs.empty()) {
            sptrobjexpr_t tobjexpr = std::static_pointer_cast<objexpr_t>(texpr);
            if (tobjexpr->obj.name.has_value()) {
                all_objs_o.push_back(tobjexpr);
            }
        }
    }); */

    auto objnamecomp = [](const sptrobjexpr_t &a, const sptrobjexpr_t &b) {
        return a->obj.name.value() < b->obj.name.value();
    };
    std::sort(all_objs_e.begin(), all_objs_e.end(), objnamecomp);
    std::sort(all_objs_o.begin(), all_objs_o.end(), objnamecomp);

    vec_t dvec;
    std::vector<std::pair<objexpr_t*, objexpr_t*>> ao;
    ao.reserve(all_objs_e.size() + all_objs_o.size());
    dvec.v.resize(all_objs_e.size() + all_objs_o.size());
    std::uint32_t i = 0, j = 0;
    while (i < all_objs_e.size() && j < all_objs_o.size()) {
        sptrobjexpr_t &teo = all_objs_e[i++];
        sptrobjexpr_t &too = all_objs_o[j++];
        if (teo == too) {
            ao.emplace_back(teo.get(), too.get());
        } else {
            if (teo->obj.name.value() < too->obj.name.value()) {
                ao.emplace_back(teo.get(), nullptr);
                j--;
            } else {
                ao.emplace_back(nullptr, too.get());
                i--;
            }
        }
    }
    if (i < all_objs_e.size()) {
        for (; i < all_objs_e.size(); i++) {
            ao.emplace_back(all_objs_e[i].get(), nullptr);
        }
    } else if (j < all_objs_o.size()) {
        for (; j < all_objs_o.size(); j++) {
            ao.emplace_back(nullptr, all_objs_o[j].get());
        }
    }

    bucket_avg_t<double> buck;
    for (std::size_t j = 0; j < sample_size; j++) {
        for (auto &[teo, too] : ao) {
            if (teo) {
                sptrobjexpr_t uo = from_expr<objexpr_t>(teo->generate());
                teo->obj = uo->obj;
                teo->signal_dirty();
            }
            if (too) {
                sptrobjexpr_t uo = from_expr<objexpr_t>(too->generate());
                too->obj = uo->obj;
                too->signal_dirty();
            }
        }
        std::optional<sptrexpr_t> re = e->calculate();
        if (re.has_value()) {
            std::optional<sptrexpr_t> ro = o->calculate();
            if (ro.has_value()) {
                sptrobjexpr_t reo = from_expr<objexpr_t>(std::move(re.value()));
                sptrobjexpr_t roo = from_expr<objexpr_t>(std::move(ro.value()));
                if (reo->obj.type != roo->obj.type) {
                    buck.add(1.0);
                } else {
                    buck.add(reo->obj_dist(*roo));
                }
            }
        }
    }

    return buck.avg_or(1.0);
}

double expr_t::dist_match(const expr_t &other) const {
    std::vector<std::vector<etype_t>> stacks, ostacks;
    std::vector<etype_t> tmp_cur, otmp_cur;
    get_deepest_type_stacks(stacks, tmp_cur);
    other.get_deepest_type_stacks(ostacks, otmp_cur);

    bucket_avg_t<double> buck{.s = 0, .c = stacks.size() * ostacks.size()};
    for (const std::vector<etype_t> &tstack : stacks) {
        for (const std::vector<etype_t> &ostack : ostacks) {
            buck.s += vec_similarity(tstack, ostack);
        }
    }

    return buck.avg_or(1.0);
}

double expr_t::dist(const expr_t &other, std::size_t sample_size = randgen_default_sample_size) const {
    double m = dist_match(other);
    double rd = dist_randgen(other, sample_size);
    return std::sqrt(m * m + rd * rd);
}


enum struct ttype_t : std::uint16_t {
    none,
    symbol_indicator,
    unresolvable_indicator,
    and_indicator,
    or_indicator,
    not_indicator,
    in_indicator,
    subset_indicator,
    union_indicator,
    intersect_indicator,
    subtract_indicator,
    st_indicator,
    map_indicator,

    comma,
    paren_open,
    paren_close,
    curly_open,
    curly_close,
    sqbrk_open,
    sqbrk_close,
    bar,
    arrow,
    colon,
    eqsign,
    exclamation,
    gtsign,
    ltsign,

    plus,
    minus,
    star,
    slash, /* forward slash */
    caret,

    other_indicator,
    name,

    mpz,
    mpfr,
};

const std::unordered_map<std::wstring, ttype_t> ttype_ind_map = { /* NOLINT */
    {L"\\symbol", ttype_t::symbol_indicator},
    {L"\\unresolvable", ttype_t::unresolvable_indicator},
    {L"\\and", ttype_t::and_indicator},
    {L"\\or", ttype_t::or_indicator},
    {L"\\not", ttype_t::not_indicator},
    {L"\\in", ttype_t::in_indicator},
    {L"\\subset", ttype_t::subset_indicator},
    {L"\\union", ttype_t::union_indicator},
    {L"\\inters", ttype_t::intersect_indicator},
    {L"\\minus", ttype_t::subtract_indicator},
    {L"\\st", ttype_t::st_indicator},
    {L"\\map", ttype_t::map_indicator},

    /* {L"", ttype_t::none}, */

    /* {L"", ttype_t::other_indicator}, */

    /* {L"", ttype_t::name} */
};

const std::unordered_map<wchar_t, ttype_t> ttype_char_map = { /* NOLINT */
    {L',', ttype_t::comma},
    {L'(', ttype_t::paren_open},
    {L')', ttype_t::paren_close},
    {L'{', ttype_t::curly_open},
    {L'}', ttype_t::curly_close},
    {L'[', ttype_t::sqbrk_open},
    {L']', ttype_t::sqbrk_close},
    {L'|', ttype_t::bar},
    {L':', ttype_t::colon},
    {L'=', ttype_t::eqsign},
    {L'!', ttype_t::exclamation},
    {L'>', ttype_t::gtsign},
    {L'<', ttype_t::ltsign},

    {L'+', ttype_t::plus},
    {L'-', ttype_t::minus},
    {L'*', ttype_t::star},
    {L'/', ttype_t::slash},
    {L'^', ttype_t::caret},
};

std::optional<std::wstring> get_key(const std::unordered_map<std::wstring, ttype_t> &m, const ttype_t &v) {
    for (const auto &[a, b] : m) {
        if (b == v) {
            return std::make_optional(a);
        }
    }
    return {};
}

template <typename T>
std::optional<T> get_value_by_view(const std::unordered_map<std::wstring, T> &m, const std::wstring_view &v) {
    for (const auto &[a, b] : m) {
        if (a == v) {
            return std::make_optional(b);
        }
    }
    return {};
}

template <typename T>
std::optional<T> get_key(const std::unordered_map<T, ttype_t> &m, const ttype_t &v) {
    for (const auto &[a, b] : m) {
        if (b == v) {
            return std::make_optional(a);
        }
    }
    return {};
}

struct parsed_tok_t {
    ttype_t type = ttype_t::none;
    std::wstring_view t;

    bool empty() const {
        return type == ttype_t::none;
    }
};


struct parser_t {
    /* everything else not in here has higher precedence! */
    static std::unordered_map<ttype_t, std::uint16_t> pred;

    /* returns true if ok, assumes pos is initially valid */
    bool get_token(const std::wstring_view &s, std::wstring_view::const_iterator &pos, std::vector<parsed_tok_t> &tbuf) {
        while (std::iswspace(*pos)) {
            pos++;
            if (pos == s.end()) {
                return true;
            }
        }
        if (*pos == L'\\') {
            if (!tbuf.empty()) {
                if (tbuf.back().type == ttype_t::other_indicator) {
                    return false;
                }
            }
            std::wstring_view::const_iterator old_pos = pos;
            pos++;
            if (pos == s.end()) {
                tbuf.push_back(parsed_tok_t{ttype_t::other_indicator, std::wstring_view(pos - 1, pos - 1)});
                return true;
            }
            if (!get_token(s, pos, tbuf)) {
                return false;
            }
            tbuf.back().t = std::wstring_view(old_pos, tbuf.back().t.end()); /* include initial backslash */
            if (std::optional<ttype_t> v = get_value_by_view(ttype_ind_map, tbuf.back().t); v.has_value()) {
                tbuf.back().type = v.value();
            } else {
                tbuf.back().type = ttype_t::other_indicator;
            }
            return true;
        } else if (std::iswdigit(*pos) || *pos == L'.' || *pos == L'-') {
            std::wstring_view::const_iterator end = pos;
            if (*end == L'-') {
                end++;
                if (end == s.end()) {
                    tbuf.push_back(parsed_tok_t{ttype_t::minus, std::wstring_view(pos, pos + 1)});
                    return true;
                }
            }
            bool seen_dot = *pos == L'.';
            bool seen_e = false;
            bool seen_e_neg = false;
            for (; end != s.end(); end++) {
                if (std::iswdigit(*end)) {
                    continue;
                }
                if (*end == L'-') {
                    if (!seen_e) {
                        break;
                    }
                    if (seen_e_neg) {
                        break;
                    } else {
                        seen_e_neg = true;
                        continue;
                    }
                }

                if (*end == L'e') {
                    if (seen_e) {
                        break;
                    }
                    seen_e = true;
                    continue;
                }

                if (*end == L'.') {
                    if (seen_dot || seen_e) {
                        break;
                    }
                    seen_dot = true;
                    continue;
                }

                break;
            }
            if (end == pos + 1 && *pos == L'-') {
                tbuf.push_back(parsed_tok_t{ttype_t::minus, std::wstring_view(pos, pos + 1)});
            } else if (seen_dot || seen_e) {
                tbuf.push_back(parsed_tok_t{ttype_t::mpfr, std::wstring_view(pos, end)});
            } else {
                tbuf.push_back(parsed_tok_t{ttype_t::mpz, std::wstring_view(pos, end)});
            }
            return true;
        } else if (ttype_char_map.contains(*pos)) {
            tbuf.push_back(parsed_tok_t{ttype_char_map.at(*pos), std::wstring_view(pos, pos + 1)});
            pos++;
            return true;
        } else { /* is name */
            std::wstring_view::const_iterator name = pos;
            for (; name != s.end(); name++) {
                if (std::iswspace(*name) || ttype_char_map.contains(*name)) {
                    break;
                }
            }
            tbuf.push_back(parsed_tok_t{ttype_t::name, std::wstring_view(pos, name)});
            pos = name;
            return true;
        }
    }

    /* if an until_tok is hit, we also consume it and place in tbuf */
    void parse_until_any_of(sptrexpr_t e, const std::wstring_view &s, const std::vector<ttype_t> &until_tok, std::optional<std::size_t> max_count = {}) { /* NOLINT */
        if (s.empty()) {
            return;
        }
        std::vector<parsed_tok_t> tbuf;
        std::wstring_view::const_iterator pos = s.begin();
        while (true) {
            if (e->exprs.size() >= max_count) {
                break;
            }
            if (!get_token(s, pos, tbuf)) {
                ERR_EXIT(1, "could not get valid token in %s", wstr_to_str(std::wstring(std::wstring_view(pos, s.end()))).c_str());
            }
            if (tbuf.empty()) {
                return;
            }
            if (tbuf.back().type == ttype_t::other_indicator) {
                ERR_EXIT(1, "could not get valid token, unfinished indicator in %s", wstr_to_str(std::wstring(std::wstring_view(pos, s.end()))).c_str())
            }
            for (const ttype_t &ut : until_tok) {
                if (ut == tbuf.back().type) {
                    return;
                }
            }
            switch (tbuf.back().type) {
                case ttype_t::symbol_indicator: {
                    if (!get_token(s, pos, tbuf)) {
                        ERR_EXIT(1, "could not get valid token, expected symbol name after \"\\symbol\" in %s", wstr_to_str(std::wstring(std::wstring_view(pos, s.end()))).c_str());
                    }
                    sptrobjexpr_t oe = make_objexpr();
                    oe->obj.type = otype_t::symbol_t;
                    oe->obj.name = tbuf.back().t;
                    e->add(oe);
                    tbuf.pop_back();
                    tbuf.pop_back();
                    break;
                }
                case ttype_t::unresolvable_indicator: {
                    sptrobjexpr_t oe = make_objexpr();
                    oe->obj.type = otype_t::unresolvable_t;
                    e->add(oe);
                    tbuf.pop_back();
                    break;
                }
                case ttype_t::not_indicator: {
                    sptrnotexpr_t ne = make_notexpr();
                    parse_until_any_of(to_expr(ne), std::wstring_view(tbuf.back().t.end(), s.end()), {}, 1);
                    e->add(ne);
                    tbuf.pop_back();
                    break;
                }
                case ttype_t::in_indicator: {
                    sptrcontainedexpr_t ce = make_containedexpr();
                    AAAAA;
                }

                /* infix section */
                case ttype_t::and_indicator: {

                }
            }
        }
    }
};



int main() {
    gmp_randinit_default(randstate);
    std::setlocale(LC_ALL, "");

    parser_t::pred = {
        {ttype_t::plus, 10},
        {ttype_t::minus, 10},
        {ttype_t::star, 10},
        {ttype_t::slash, 10},
        {ttype_t::caret, 10},
        {ttype_t::bar, 50},
        {ttype_t::comma, 100},
    };

    // void *dlhandle = dlopen(WORK_SO_FILENAME, RTLD_LAZY);
    // expr_work = reinterpret_cast<decltype(expr_work)>(dlsym(dlhandle, "expr_work")); /* NOLINT */
    // if (expr_work == nullptr) {
    //     ERR_EXIT(1, "could not resolve expr_work from shared object " WORK_SO_FILENAME);
    // }
    // dlclose(dlhandle);

    /* define f: {} -> {} := x -> x + 1 */
    sptrmapdefexpr_t fndef = make_mapdefexpr();
    sptrobjexpr_t domainset = make_objexpr();
    domainset->obj.type = otype_t::set_t;
    fndef->add(domainset->copy());
    fndef->add(domainset);
    sptraddexpr_t fnbody = make_addexpr();
    sptrnegexpr_t neg = make_negexpr();
    sptrobjexpr_t var = make_objexpr();
    var->obj.name = L"x";
    neg->add(var);
    fnbody->add(neg);
    mpzw_t i{};
    mpz_init_set_ui(i.z, 1);
    sptrobjexpr_t one = make_mpzw(i);
    fnbody->add(one);
    fndef->add(fnbody);
    fndef->func_name = L"f";
    fndef->param_names.emplace_back(L"x");
    expr_add_to_defs(fndef);

    /* call the function */
    sptrmapcallexpr_t fncall = from_expr<mapcallexpr_t>(make_mapcallexpr());
    sptrobjexpr_t fnref = make_objexpr();
    fnref->obj.name = L"f";
    fnref->obj.type = otype_t::map_t;
    fncall->add(fnref);
    fncall->add(one->copy());

    sptrobjexpr_t ores = from_expr<objexpr_t>(fncall->calculate().value_or(make_objexpr()));
    std::wcout << "defined ";
    fndef->disp(std::wcout, {});
    std::wcout << std::endl;
    fncall->disp(std::wcout, {});
    std::wcout << " loaded to ";
    ores->disp(std::wcout, {});
    std::wcout << std::endl;

    gmp_randclear(randstate);
}
