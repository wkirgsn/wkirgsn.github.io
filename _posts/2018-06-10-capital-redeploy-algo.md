---
layout: post
title: "Capital Redeployment Algorithm"
date: 2018-06-10
comments: true
---

Some capital investment experts tend to say there are two basic things that will lead to a high-performing depot:

* A risk-balanced, diversified portfolio and
* Periodic asset redeployment

The former has no definitive answer and should be adapted to each investor individually.
Generally speaking, higher-risk assets can take up a larger amount of younger money-savers' total investment since they have more time to put up for adverse market trends whereas the older generation should stick to steadier securities. For a good overview of different portfolio types see [Investopedia's guideline on popuular portfolio types](https://www.investopedia.com/articles/basics/11/5-popular-portfolio-types.asp). However, diversification is comfortably achieved by passively managed funds as, for instance, ETFs, which come with large cost savings and outperform actively managed funds almost all the time - [the Economist confirms](https://www.economist.com/finance-and-economics/2017/06/24/fund-managers-rarely-outperform-the-market-for-long).

The second virtue for a successful investment into stocks on the other hand is dead simple:
Reassess your capital as often as you feel comfortable with, at least once a year or as soon as your asset weightings fall apart drastically, say 5% in one asset weighting.

# What is Asset Redeployment?
Reassessing capital means one would transfer money on all her/his different assets in such a way, that each asset holds an amount equal to the weighting you initially assigned to it.

Imagine for example you like 30% of your capital in stocks and the other 70% in bonds since that is a risk-apportionment you can live with.
The next day stock market booms and you see your stock capital soaring up to undreamed heights.
The first moment you jump for joy but the next you halt as one glance on the new capital weights makes you shiver: stocks and bonds are fifty-fifty.
Your risk-portfolio is suddenly highly imbalanced and you are burning for action, there is a money transfer necessary.
You go ahead and deposit the required amount on your bond asset.

## Nothing simpler than that!
But hold on!
I don't have two assets, I have five or six!
Besides that, I don't place high figures at once on my portfolio, I do small monthly payments.

Calculating different amounts necessary for multiple assets that change in different directions is non-trivial and gave me headaches in the beginning.
Very soon it became clear, that it must be solved iteratively through an algorithm.

The following python script shows how it can be done.

# The Algorithm

```py
"""Mapping for current balance on each kind of capital and the desired 
weight. Dict with identifier keys and dict-values that have current and 
target pairs."""
CURRENT_TARGET_DICT = {
    'Stocks':
        {'actual':  6752,
         'target_weight': 0.3},
    'Commodities':
        {'actual': 3448,
         'target_weight': 0.15},
    'Long-Term Bonds':
        {'actual': 8360,
         'target_weight': 0.4},
    'Mid-Term Bonds':
        {'actual':  3089 ,
         'target_weight': 0.15}
}

# The monthly rate that is remitted and split among all accounts
MONTHLY_RATE = 500


def main():
    max_months_until_targets_met = 24
    total_capital = sum(v['actual'] for k, v in CURRENT_TARGET_DICT.items())

    redeployment_split_plan = {}
    # find amount of months until portfolio targets met
    for _m in range(1, max_months_until_targets_met+1):
        further_increase_m = False
        for acc, current_dict in CURRENT_TARGET_DICT.items():
            rate = \
                (total_capital+_m*MONTHLY_RATE)*current_dict['target_weight']\
                - current_dict['actual']
            if abs(rate) <= _m*MONTHLY_RATE and not rate < 0:
                # current amount of months is feasible for this account
                redeployment_split_plan[acc] = {'total_required': rate}
                continue
            else:
                # too few months to meet portfolio targets
                further_increase_m = True
                break

        if further_increase_m:
            continue
        else:
            total_remit = sum(v['total_required'] for k, v in
                              redeployment_split_plan.items())
            required_months = int(total_remit // MONTHLY_RATE)
            # calculate schedule
            for acc, plan in redeployment_split_plan.items():
                rate = \
                    (plan['total_required'] / total_remit)*MONTHLY_RATE

                plan['schedule'] = [rate, ]*required_months

            # schedule calculation finished for all accounts
            break
    else:
        raise ValueError('ERR: No schedule found within the next {} months'
                         .format(max_months_until_targets_met))

    report_schedule(redeployment_split_plan, total_capital)


def report_schedule(plan, _total_capital):
    print('## Redeployment Plan ##\n')
    for acc, info in plan.items():
        print(acc)
        print('Current ratio: {:.3f} - target: {:.3f}'
              .format(CURRENT_TARGET_DICT[acc]['actual']/_total_capital,
                      CURRENT_TARGET_DICT[acc]['target_weight']))
        print('Total Required: {:.2f} EUR'.format(info['total_required']))
        print('Remit {:.2f} EUR for {:} months'.format(info['schedule'][0],
                                                  len(info['schedule'])))
        print('After that, continue with a rate of {:.2f} EUR\n'.format(
            MONTHLY_RATE*CURRENT_TARGET_DICT[acc]['target_weight']))


if __name__ == '__main__':
    main()
```

The user only has to adapt the *CURRENT_TARGET_DICT* and *MONTHLY_RATE* in the beginning of the script to her/his own investment situation.
With the dummy values that are shown in the script, the following output is to be expected:

```
## Redeployment Plan ##

Stocks
Current ratio: 0.312 - target: 0.300
Total Required: 192.70 EUR
Remit 64.23 EUR for 3 months
After that, continue with a rate of 150.00 EUR

Long-Term Bonds
Current ratio: 0.386 - target: 0.400
Total Required: 899.60 EUR
Remit 299.87 EUR for 3 months
After that, continue with a rate of 200.00 EUR

Commodities
Current ratio: 0.159 - target: 0.150
Total Required: 24.35 EUR
Remit 8.12 EUR for 3 months
After that, continue with a rate of 75.00 EUR

Mid-Term Bonds
Current ratio: 0.143 - target: 0.150
Total Required: 383.35 EUR
Remit 127.78 EUR for 3 months
After that, continue with a rate of 75.00 EUR
```
## Why not withdrawing from assets with too much weight?
The general approach in reassing the asset weights is by remitting positve amounts on each asset in an appropriated ratio.
Reason for this is that withdrawing money from funds can lead to taxable profit, which would impair your earnings as the current point of time for reassing assets is seldomly the correct moment to sell shares with regards to the market course.


## Mathematical Background
The approach is straight forward and will be illustrated with two hypothetical assets.
With x1 and x2 being the unknown rates to remit to asset one and two, m being the unknown amount of months required, B being the total monthly rate remittable while S1 and S2 being the percentual target share for asset one and two and having K1 and K2 as current balances, we'll have the following equations:

    x1 + x2 = m * B
    K1 + x1 = (K1 + K2 + m*B) * S1
    K2 + x2 = (K1 + K2 + m*B) * S2
    
    <=> x1 = (K1 + K2 + m*B) * S1 - K1  (1)
    <=> x2 = (K1 + K2 + m*B) * S2 - K2  (2)
    
    0 <= x1 <= m * B    (3)
    0 <= x2 <= m * B    (4)

With eq. (1) and (2) we have the formulas for the amount to remit to each asset as soon as we have a value for the unknown _m_ (number of months).
Since negative rates are not allowed, the algorithm will count up _m_ and attempt to get non-negative rates for each asset until a proper amount of months is found.

# Conclusion
Every once or twice a year, when it comes down to reassing your portfolio, you can help yourself with this script in order to find the required values for remittance such that your risk profile is healthy again.

Please feel free to convert this algorithm to your preferred programming language.

If you want to read more on successfully building up a portfolio you can follow up on [this Investopedia article](https://www.investopedia.com/financial-advisor/steps-building-profitable-portfolio/).

* * *
