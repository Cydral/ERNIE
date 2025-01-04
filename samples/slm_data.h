#ifndef SlmData_H
#define SlmData_H

#include <string>
#include <vector>
#include <algorithm>

// Utility function to concatenate text parts
std::string concatenateTexts(const std::vector<std::string>& texts) {
    std::string result;
    for (const auto& text : texts) {
        result += text;
    }
    return result;
}

// Text parts for training
const std::vector<std::string> shakespeare_text_parts = {
    R"(QUEEN ELIZABETH:
Go, go, poor soul, I envy not thy glory
To feed my humour, wish thyself no harm.

LADY ANNE:
No! why?  When he that is my husband now
Came to me, as I follow'd Henry's corse,
When scarce the blood was well wash'd from his hands
Which issued from my other angel husband
And that dead saint which then I weeping follow'd;
O, when, I say, I look'd on Richard's face,
This was my wish: 'Be thou,' quoth I, ' accursed,
For making me, so young, so old a widow!
And, when thou wed'st, let sorrow haunt thy bed;
And be thy wife--if any be so mad--
As miserable by the life of thee
As thou hast made me by my dear lord's death!
Lo, ere I can repeat this curse again,
Even in so short a space, my woman's heart
Grossly grew captive to his honey words
And proved the subject of my own soul's curse,
Which ever since hath kept my eyes from rest;
For never yet one hour in his bed
Have I enjoy'd the golden dew of sleep,
But have been waked by his timorous dreams.
Besides, he hates me for my father Warwick;
And will, no doubt, shortly be rid of me.

QUEEN ELIZABETH:
Poor heart, adieu! I pity thy complaining.

LADY ANNE:
No more than from my soul I mourn for yours.

QUEEN ELIZABETH:
Farewell, thou woful welcomer of glory!

LADY ANNE:
Adieu, poor soul, that takest thy leave of it!

DUCHESS OF YORK:

QUEEN ELIZABETH:
Stay, yet look back with me unto the Tower.
Pity, you ancient stones, those tender babes
Whom envy hath immured within your walls!
Rough cradle for such little pretty ones!
Rude ragged nurse, old sullen playfellow
For tender princes, use my babies well!
So foolish sorrow bids your stones farewell.

KING RICHARD III:
Stand all apart Cousin of Buckingham!

BUCKINGHAM:
My gracious sovereign?

KING RICHARD III:
Give me thy hand.
Thus high, by thy advice
And thy assistance, is King Richard seated;
But shall we wear these honours for a day?
Or shall they last, and we rejoice in them?

BUCKINGHAM:
Still live they and for ever may they last!

KING RICHARD III:
O Buckingham, now do I play the touch,
To try if thou be current gold indeed
Young Edward lives: think now what I would say.

BUCKINGHAM:
Say on, my loving lord.

KING RICHARD III:
Why, Buckingham, I say, I would be king,

BUCKINGHAM:
Why, so you are, my thrice renowned liege.

KING RICHARD III:
Ha! am I king? 'tis so: but Edward lives.

BUCKINGHAM:
True, noble prince.

KING RICHARD III:
O bitter consequence,
That Edward still should live! 'True, noble prince!'
Cousin, thou wert not wont to be so dull:
Shall I be plain? I wish the bastards dead;
And I would have it suddenly perform'd.
What sayest thou? speak suddenly; be brief.

BUCKINGHAM:
Your grace may do your pleasure.

KING RICHARD III:
Tut, tut, thou art all ice, thy kindness freezeth:
Say, have I thy consent that they shall die?

BUCKINGHAM:
Give me some breath, some little pause, my lord
Before I positively herein:
I will resolve your grace immediately.

)",

    R"(KING RICHARD III:
I will converse with iron-witted fools
And unrespective boys: none are for me
That look into me with considerate eyes:
High-reaching Buckingham grows circumspect.
Boy!

Page:
My lord?

KING RICHARD III:
Know'st thou not any whom corrupting gold
Would tempt unto a close exploit of death?

Page:
My lord, I know a discontented gentleman,
Whose humble means match not his haughty mind:
Gold were as good as twenty orators,
And will, no doubt, tempt him to any thing.

KING RICHARD III:
What is his name?

Page:
His name, my lord, is Tyrrel.

KING RICHARD III:
I partly know the man: go, call him hither.
The deep-revolving witty Buckingham
No more shall be the neighbour to my counsel:
Hath he so long held out with me untired,
And stops he now for breath?
How now! what news with you?

STANLEY:
My lord, I hear the Marquis Dorset's fled
To Richmond, in those parts beyond the sea
Where he abides.

KING RICHARD III:
Catesby!

CATESBY:
My lord?

KING RICHARD III:
Rumour it abroad
That Anne, my wife, is sick and like to die:
I will take order for her keeping close.
Inquire me out some mean-born gentleman,
Whom I will marry straight to Clarence' daughter:
The boy is foolish, and I fear not him.
Look, how thou dream'st! I say again, give out
That Anne my wife is sick and like to die:
About it; for it stands me much upon,
To stop all hopes whose growth may damage me.
I must be married to my brother's daughter,
Or else my kingdom stands on brittle glass.
Murder her brothers, and then marry her!
Uncertain way of gain! But I am in
So far in blood that sin will pluck on sin:
Tear-falling pity dwells not in this eye.
Is thy name Tyrrel?

TYRREL:
James Tyrrel, and your most obedient subject.

KING RICHARD III:
Art thou, indeed?

TYRREL:
Prove me, my gracious sovereign.

KING RICHARD III:
Darest thou resolve to kill a friend of mine?

TYRREL:
Ay, my lord;
But I had rather kill two enemies.

KING RICHARD III:
Why, there thou hast it: two deep enemies,
Foes to my rest and my sweet sleep's disturbers
Are they that I would have thee deal upon:
Tyrrel, I mean those bastards in the Tower.

TYRREL:
Let me have open means to come to them,
And soon I'll rid you from the fear of them.

KING RICHARD III:
Thou sing'st sweet music. Hark, come hither, Tyrrel
Go, by this token: rise, and lend thine ear:
There is no more but so: say it is done,
And I will love thee, and prefer thee too.

TYRREL:
'Tis done, my gracious lord.

KING RICHARD III:
Shall we hear from thee, Tyrrel, ere we sleep?

)",

    R"(TYRREL:
Ye shall, my Lord.

BUCKINGHAM:
My Lord, I have consider'd in my mind
The late demand that you did sound me in.

KING RICHARD III:
Well, let that pass. Dorset is fled to Richmond.

BUCKINGHAM:
I hear that news, my lord.

KING RICHARD III:
Stanley, he is your wife's son well, look to it.

BUCKINGHAM:
My lord, I claim your gift, my due by promise,
For which your honour and your faith is pawn'd;
The earldom of Hereford and the moveables
The which you promised I should possess.

KING RICHARD III:
Stanley, look to your wife; if she convey
Letters to Richmond, you shall answer it.

BUCKINGHAM:
What says your highness to my just demand?

KING RICHARD III:
As I remember, Henry the Sixth
Did prophesy that Richmond should be king,
When Richmond was a little peevish boy.
A king, perhaps, perhaps,--

BUCKINGHAM:
My lord!

KING RICHARD III:
How chance the prophet could not at that time
Have told me, I being by, that I should kill him?

BUCKINGHAM:
My lord, your promise for the earldom,--

KING RICHARD III:
Richmond! When last I was at Exeter,
The mayor in courtesy show'd me the castle,
And call'd it Rougemont: at which name I started,
Because a bard of Ireland told me once
I should not live long after I saw Richmond.

BUCKINGHAM:
My Lord!

KING RICHARD III:
Ay, what's o'clock?

BUCKINGHAM:
I am thus bold to put your grace in mind
Of what you promised me.

KING RICHARD III:
Well, but what's o'clock?

BUCKINGHAM:
Upon the stroke of ten.

KING RICHARD III:
Well, let it strike.

BUCKINGHAM:
Why let it strike?

KING RICHARD III:
Because that, like a Jack, thou keep'st the stroke
Betwixt thy begging and my meditation.
I am not in the giving vein to-day.

BUCKINGHAM:
Why, then resolve me whether you will or no.

KING RICHARD III:
Tut, tut,
Thou troublest me; am not in the vein.

BUCKINGHAM:
Is it even so? rewards he my true service
With such deep contempt made I him king for this?
O, let me think on Hastings, and be gone
To Brecknock, while my fearful head is on!

)",

    R"(TYRREL:
The tyrannous and bloody deed is done.
The most arch of piteous massacre
That ever yet this land was guilty of.
Dighton and Forrest, whom I did suborn
To do this ruthless piece of butchery,
Although they were flesh'd villains, bloody dogs,
Melting with tenderness and kind compassion
Wept like two children in their deaths' sad stories.
'Lo, thus' quoth Dighton, 'lay those tender babes:'
'Thus, thus,' quoth Forrest, 'girdling one another
Within their innocent alabaster arms:
Their lips were four red roses on a stalk,
Which in their summer beauty kiss'd each other.
A book of prayers on their pillow lay;
Which once,' quoth Forrest, 'almost changed my mind;
But O! the devil'--there the villain stopp'd
Whilst Dighton thus told on: 'We smothered
The most replenished sweet work of nature,
That from the prime creation e'er she framed.'
Thus both are gone with conscience and remorse;
They could not speak; and so I left them both,
To bring this tidings to the bloody king.
And here he comes.
All hail, my sovereign liege!

KING RICHARD III:
Kind Tyrrel, am I happy in thy news?

TYRREL:
If to have done the thing you gave in charge
Beget your happiness, be happy then,
For it is done, my lord.

KING RICHARD III:
But didst thou see them dead?

TYRREL:
I did, my lord.

KING RICHARD III:
And buried, gentle Tyrrel?

TYRREL:
The chaplain of the Tower hath buried them;
But how or in what place I do not know.

KING RICHARD III:
Come to me, Tyrrel, soon at after supper,
And thou shalt tell the process of their death.
Meantime, but think how I may do thee good,
And be inheritor of thy desire.
Farewell till soon.
The son of Clarence have I pent up close;
His daughter meanly have I match'd in marriage;
The sons of Edward sleep in Abraham's bosom,
And Anne my wife hath bid the world good night.
Now, for I know the Breton Richmond aims
At young Elizabeth, my brother's daughter,
And, by that knot, looks proudly o'er the crown,
To her I go, a jolly thriving wooer.

CATESBY:
My lord!

KING RICHARD III:
Good news or bad, that thou comest in so bluntly?

CATESBY:
Bad news, my lord: Ely is fled to Richmond;
And Buckingham, back'd with the hardy Welshmen,
Is in the field, and still his power increaseth.

KING RICHARD III:
Ely with Richmond troubles me more near
Than Buckingham and his rash-levied army.
Come, I have heard that fearful commenting
Is leaden servitor to dull delay;
Delay leads impotent and snail-paced beggary
Then fiery expedition be my wing,
Jove's Mercury, and herald for a king!
Come, muster men: my counsel is my shield;
We must be brief when traitors brave the field.

QUEEN MARGARET:
So, now prosperity begins to mellow
And drop into the rotten mouth of death.
Here in these confines slily have I lurk'd,
To watch the waning of mine adversaries.
A dire induction am I witness to,
And will to France, hoping the consequence
Will prove as bitter, black, and tragical.
Withdraw thee, wretched Margaret: who comes here?

QUEEN ELIZABETH:
Ah, my young princes! ah, my tender babes!
My unblown flowers, new-appearing sweets!
If yet your gentle souls fly in the air
And be not fix'd in doom perpetual,
Hover about me with your airy wings
And hear your mother's lamentation!

QUEEN MARGARET:
Hover about her; say, that right for right
Hath dimm'd your infant morn to aged night.

DUCHESS OF YORK:
So many miseries have crazed my voice,
That my woe-wearied tongue is mute and dumb,
Edward Plantagenet, why art thou dead?

QUEEN MARGARET:
Plantagenet doth quit Plantagenet.
Edward for Edward pays a dying debt.

QUEEN ELIZABETH:
Wilt thou, O God, fly from such gentle lambs,
And throw them in the entrails of the wolf?
When didst thou sleep when such a deed was done?

QUEEN MARGARET:
When holy Harry died, and my sweet son.

DUCHESS OF YORK:
Blind sight, dead life, poor mortal living ghost,
Woe's scene, world's shame, grave's due by life usurp'd,
Brief abstract and record of tedious days,
Rest thy unrest on England's lawful earth,
Unlawfully made drunk with innocents' blood!

QUEEN ELIZABETH:
O, that thou wouldst as well afford a grave
As thou canst yield a melancholy seat!
Then would I hide my bones, not rest them here.
O, who hath any cause to mourn but I?

QUEEN MARGARET:
If ancient sorrow be most reverend,
Give mine the benefit of seniory,
And let my woes frown on the upper hand.
If sorrow can admit society,
Tell o'er your woes again by viewing mine:
I had an Edward, till a Richard kill'd him;
I had a Harry, till a Richard kill'd him:
Thou hadst an Edward, till a Richard kill'd him;
Thou hadst a Richard, till a Richard killed him;

DUCHESS OF YORK:
I had a Richard too, and thou didst kill him;
I had a Rutland too, thou holp'st to kill him.

QUEEN MARGARET:
Thou hadst a Clarence too, and Richard kill'd him.
From forth the kennel of thy womb hath crept
A hell-hound that doth hunt us all to death:
That dog, that had his teeth before his eyes,
To worry lambs and lap their gentle blood,
That foul defacer of God's handiwork,
That excellent grand tyrant of the earth,
That reigns in galled eyes of weeping souls,
Thy womb let loose, to chase us to our graves.
O upright, just, and true-disposing God,
How do I thank thee, that this carnal cur
Preys on the issue of his mother's body,
And makes her pew-fellow with others' moan!

)",

    R"(DUCHESS OF YORK:
O Harry's wife, triumph not in my woes!
God witness with me, I have wept for thine.

QUEEN MARGARET:
Bear with me; I am hungry for revenge,
And now I cloy me with beholding it.
Thy Edward he is dead, that stabb'd my Edward:
Thy other Edward dead, to quit my Edward;
Young York he is but boot, because both they
Match not the high perfection of my loss:
Thy Clarence he is dead that kill'd my Edward;
And the beholders of this tragic play,
The adulterate Hastings, Rivers, Vaughan, Grey,
Untimely smother'd in their dusky graves.
Richard yet lives, hell's black intelligencer,
Only reserved their factor, to buy souls
And send them thither: but at hand, at hand,
Ensues his piteous and unpitied end:
Earth gapes, hell burns, fiends roar, saints pray.
To have him suddenly convey'd away.
Cancel his bond of life, dear God, I prey,
That I may live to say, The dog is dead!

QUEEN ELIZABETH:
O, thou didst prophesy the time would come
That I should wish for thee to help me curse
That bottled spider, that foul bunch-back'd toad!

QUEEN MARGARET:
I call'd thee then vain flourish of my fortune;
I call'd thee then poor shadow, painted queen;
The presentation of but what I was;
The flattering index of a direful pageant;
One heaved a-high, to be hurl'd down below;
A mother only mock'd with two sweet babes;
A dream of what thou wert, a breath, a bubble,
A sign of dignity, a garish flag,
To be the aim of every dangerous shot,
A queen in jest, only to fill the scene.
Where is thy husband now? where be thy brothers?
Where are thy children? wherein dost thou, joy?
Who sues to thee and cries 'God save the queen'?
Where be the bending peers that flatter'd thee?
Where be the thronging troops that follow'd thee?
Decline all this, and see what now thou art:
For happy wife, a most distressed widow;
For joyful mother, one that wails the name;
For queen, a very caitiff crown'd with care;
For one being sued to, one that humbly sues;
For one that scorn'd at me, now scorn'd of me;
For one being fear'd of all, now fearing one;
For one commanding all, obey'd of none.
Thus hath the course of justice wheel'd about,
And left thee but a very prey to time;
Having no more but thought of what thou wert,
To torture thee the more, being what thou art.
Thou didst usurp my place, and dost thou not
Usurp the just proportion of my sorrow?
Now thy proud neck bears half my burthen'd yoke;
From which even here I slip my weary neck,
And leave the burthen of it all on thee.
Farewell, York's wife, and queen of sad mischance:
These English woes will make me smile in France.

QUEEN ELIZABETH:
O thou well skill'd in curses, stay awhile,
And teach me how to curse mine enemies!

QUEEN MARGARET:
Forbear to sleep the nights, and fast the days;
Compare dead happiness with living woe;
Think that thy babes were fairer than they were,
And he that slew them fouler than he is:
Bettering thy loss makes the bad causer worse:
Revolving this will teach thee how to curse.

QUEEN ELIZABETH:
My words are dull; O, quicken them with thine!

QUEEN MARGARET:
Thy woes will make them sharp, and pierce like mine.

DUCHESS OF YORK:
Why should calamity be full of words?

QUEEN ELIZABETH:
Windy attorneys to their client woes,
Airy succeeders of intestate joys,
Poor breathing orators of miseries!
Let them have scope: though what they do impart
Help not all, yet do they ease the heart.

DUCHESS OF YORK:
If so, then be not tongue-tied: go with me.
And in the breath of bitter words let's smother
My damned son, which thy two sweet sons smother'd.
I hear his drum: be copious in exclaims.

KING RICHARD III:
Who intercepts my expedition?

DUCHESS OF YORK:
O, she that might have intercepted thee,
By strangling thee in her accursed womb
From all the slaughters, wretch, that thou hast done!

QUEEN ELIZABETH:
Hidest thou that forehead with a golden crown,
Where should be graven, if that right were right,
The slaughter of the prince that owed that crown,
And the dire death of my two sons and brothers?
Tell me, thou villain slave, where are my children?

DUCHESS OF YORK:
Thou toad, thou toad, where is thy brother Clarence?
And little Ned Plantagenet, his son?

QUEEN ELIZABETH:
Where is kind Hastings, Rivers, Vaughan, Grey?

KING RICHARD III:
A flourish, trumpets! strike alarum, drums!
Let not the heavens hear these tell-tale women
Rail on the Lord's enointed: strike, I say!
Either be patient, and entreat me fair,
Or with the clamorous report of war
Thus will I drown your exclamations.

DUCHESS OF YORK:
Art thou my son?

KING RICHARD III:
Ay, I thank God, my father, and yourself.

DUCHESS OF YORK:
Then patiently hear my impatience.

KING RICHARD III:
Madam, I have a touch of your condition,
Which cannot brook the accent of reproof.

DUCHESS OF YORK:
O, let me speak!

KING RICHARD III:
Do then: but I'll not hear.

DUCHESS OF YORK:
I will be mild and gentle in my speech.

KING RICHARD III:
And brief, good mother; for I am in haste.

DUCHESS OF YORK:
Art thou so hasty? I have stay'd for thee,
God knows, in anguish, pain and agony.

KING RICHARD III:
And came I not at last to comfort you?

)",

    R"(DUCHESS OF YORK:
No, by the holy rood, thou know'st it well,
Thou camest on earth to make the earth my hell.
A grievous burthen was thy birth to me;
Tetchy and wayward was thy infancy;
Thy school-days frightful, desperate, wild, and furious,
Thy prime of manhood daring, bold, and venturous,
Thy age confirm'd, proud, subdued, bloody,
treacherous,
More mild, but yet more harmful, kind in hatred:
What comfortable hour canst thou name,
That ever graced me in thy company?

KING RICHARD III:
Faith, none, but Humphrey Hour, that call'd
your grace
To breakfast once forth of my company.
If I be so disgracious in your sight,
Let me march on, and not offend your grace.
Strike the drum.

DUCHESS OF YORK:
I prithee, hear me speak.

KING RICHARD III:
You speak too bitterly.

DUCHESS OF YORK:
Hear me a word;
For I shall never speak to thee again.

KING RICHARD III:
So.

DUCHESS OF YORK:
Either thou wilt die, by God's just ordinance,
Ere from this war thou turn a conqueror,
Or I with grief and extreme age shall perish
And never look upon thy face again.
Therefore take with thee my most heavy curse;
Which, in the day of battle, tire thee more
Than all the complete armour that thou wear'st!
My prayers on the adverse party fight;
And there the little souls of Edward's children
Whisper the spirits of thine enemies
And promise them success and victory.
Bloody thou art, bloody will be thy end;
Shame serves thy life and doth thy death attend.

QUEEN ELIZABETH:
Though far more cause, yet much less spirit to curse
Abides in me; I say amen to all.

KING RICHARD III:
Stay, madam; I must speak a word with you.

QUEEN ELIZABETH:
I have no more sons of the royal blood
For thee to murder: for my daughters, Richard,
They shall be praying nuns, not weeping queens;
And therefore level not to hit their lives.

KING RICHARD III:
You have a daughter call'd Elizabeth,
Virtuous and fair, royal and gracious.

QUEEN ELIZABETH:
And must she die for this? O, let her live,
And I'll corrupt her manners, stain her beauty;
Slander myself as false to Edward's bed;
Throw over her the veil of infamy:
So she may live unscarr'd of bleeding slaughter,
I will confess she was not Edward's daughter.

KING RICHARD III:
Wrong not her birth, she is of royal blood.

QUEEN ELIZABETH:
To save her life, I'll say she is not so.

KING RICHARD III:
Her life is only safest in her birth.

QUEEN ELIZABETH:
And only in that safety died her brothers.

KING RICHARD III:
Lo, at their births good stars were opposite.

QUEEN ELIZABETH:
No, to their lives bad friends were contrary.

KING RICHARD III:
All unavoided is the doom of destiny.

QUEEN ELIZABETH:
True, when avoided grace makes destiny:
My babes were destined to a fairer death,
If grace had bless'd thee with a fairer life.

KING RICHARD III:
You speak as if that I had slain my cousins.

QUEEN ELIZABETH:
Cousins, indeed; and by their uncle cozen'd
Of comfort, kingdom, kindred, freedom, life.
Whose hand soever lanced their tender hearts,
Thy head, all indirectly, gave direction:
No doubt the murderous knife was dull and blunt
Till it was whetted on thy stone-hard heart,
To revel in the entrails of my lambs.
But that still use of grief makes wild grief tame,
My tongue should to thy ears not name my boys
Till that my nails were anchor'd in thine eyes;
And I, in such a desperate bay of death,
Like a poor bark, of sails and tackling reft,
Rush all to pieces on thy rocky bosom.

KING RICHARD III:
Madam, so thrive I in my enterprise
And dangerous success of bloody wars,
As I intend more good to you and yours,
Than ever you or yours were by me wrong'd!

QUEEN ELIZABETH:
What good is cover'd with the face of heaven,
To be discover'd, that can do me good?

KING RICHARD III:
The advancement of your children, gentle lady.

QUEEN ELIZABETH:
Up to some scaffold, there to lose their heads?

KING RICHARD III:
No, to the dignity and height of honour
The high imperial type of this earth's glory.

QUEEN ELIZABETH:
Flatter my sorrows with report of it;
Tell me what state, what dignity, what honour,
Canst thou demise to any child of mine?

KING RICHARD III:
Even all I have; yea, and myself and all,
Will I withal endow a child of thine;
So in the Lethe of thy angry soul
Thou drown the sad remembrance of those wrongs
Which thou supposest I have done to thee.

QUEEN ELIZABETH:
Be brief, lest that be process of thy kindness
Last longer telling than thy kindness' date.

KING RICHARD III:
Then know, that from my soul I love thy daughter.

QUEEN ELIZABETH:
My daughter's mother thinks it with her soul.

KING RICHARD III:
What do you think?

QUEEN ELIZABETH:
That thou dost love my daughter from thy soul:
So from thy soul's love didst thou love her brothers;
And from my heart's love I do thank thee for it.

KING RICHARD III:
Be not so hasty to confound my meaning:
I mean, that with my soul I love thy daughter,
And mean to make her queen of England.

QUEEN ELIZABETH:
Say then, who dost thou mean shall be her king?

KING RICHARD III:
Even he that makes her queen who should be else?

QUEEN ELIZABETH:
What, thou?

KING RICHARD III:
I, even I: what think you of it, madam?

QUEEN ELIZABETH:
How canst thou woo her?

KING RICHARD III:
That would I learn of you,
As one that are best acquainted with her humour.

QUEEN ELIZABETH:
And wilt thou learn of me?

KING RICHARD III:
Madam, with all my heart.

)",

    R"(QUEEN ELIZABETH:
Send to her, by the man that slew her brothers,
A pair of bleeding-hearts; thereon engrave
Edward and York; then haply she will weep:
Therefore present to her--as sometime Margaret
Did to thy father, steep'd in Rutland's blood,--
A handkerchief; which, say to her, did drain
The purple sap from her sweet brother's body
And bid her dry her weeping eyes therewith.
If this inducement force her not to love,
Send her a story of thy noble acts;
Tell her thou madest away her uncle Clarence,
Her uncle Rivers; yea, and, for her sake,
Madest quick conveyance with her good aunt Anne.

KING RICHARD III:
Come, come, you mock me; this is not the way
To win our daughter.

QUEEN ELIZABETH:
There is no other way
Unless thou couldst put on some other shape,
And not be Richard that hath done all this.

KING RICHARD III:
Say that I did all this for love of her.

QUEEN ELIZABETH:
Nay, then indeed she cannot choose but hate thee,
Having bought love with such a bloody spoil.

KING RICHARD III:
Look, what is done cannot be now amended:
Men shall deal unadvisedly sometimes,
Which after hours give leisure to repent.
If I did take the kingdom from your sons,
To make amends, Ill give it to your daughter.
If I have kill'd the issue of your womb,
To quicken your increase, I will beget
Mine issue of your blood upon your daughter
A grandam's name is little less in love
Than is the doting title of a mother;
They are as children but one step below,
Even of your mettle, of your very blood;
Of an one pain, save for a night of groans
Endured of her, for whom you bid like sorrow.
Your children were vexation to your youth,
But mine shall be a comfort to your age.
The loss you have is but a son being king,
And by that loss your daughter is made queen.
I cannot make you what amends I would,
Therefore accept such kindness as I can.
Dorset your son, that with a fearful soul
Leads discontented steps in foreign soil,
This fair alliance quickly shall call home
To high promotions and great dignity:
The king, that calls your beauteous daughter wife.
Familiarly shall call thy Dorset brother;
Again shall you be mother to a king,
And all the ruins of distressful times
Repair'd with double riches of content.
What! we have many goodly days to see:
The liquid drops of tears that you have shed
Shall come again, transform'd to orient pearl,
Advantaging their loan with interest
Of ten times double gain of happiness.
Go, then my mother, to thy daughter go
Make bold her bashful years with your experience;
Prepare her ears to hear a wooer's tale
Put in her tender heart the aspiring flame
Of golden sovereignty; acquaint the princess
With the sweet silent hours of marriage joys
And when this arm of mine hath chastised
The petty rebel, dull-brain'd Buckingham,
Bound with triumphant garlands will I come
And lead thy daughter to a conqueror's bed;
To whom I will retail my conquest won,
And she shall be sole victress, Caesar's Caesar.

QUEEN ELIZABETH:
What were I best to say? her father's brother
Would be her lord? or shall I say, her uncle?
Or, he that slew her brothers and her uncles?
Under what title shall I woo for thee,
That God, the law, my honour and her love,
Can make seem pleasing to her tender years?

KING RICHARD III:
Infer fair England's peace by this alliance.

QUEEN ELIZABETH:
Which she shall purchase with still lasting war.

KING RICHARD III:
Say that the king, which may command, entreats.

QUEEN ELIZABETH:
That at her hands which the king's King forbids.

KING RICHARD III:
Say, she shall be a high and mighty queen.

QUEEN ELIZABETH:
To wail the tide, as her mother doth.

KING RICHARD III:
Say, I will love her everlastingly.

QUEEN ELIZABETH:
But how long shall that title 'ever' last?

KING RICHARD III:
Sweetly in force unto her fair life's end.

QUEEN ELIZABETH:
But how long fairly shall her sweet lie last?

KING RICHARD III:
So long as heaven and nature lengthens it.

QUEEN ELIZABETH:
So long as hell and Richard likes of it.

KING RICHARD III:
Say, I, her sovereign, am her subject love.

QUEEN ELIZABETH:
But she, your subject, loathes such sovereignty.

KING RICHARD III:
Be eloquent in my behalf to her.

QUEEN ELIZABETH:
An honest tale speeds best being plainly told.

KING RICHARD III:
Then in plain terms tell her my loving tale.

QUEEN ELIZABETH:
Plain and not honest is too harsh a style.

)",

    R"(KING RICHARD III:
Your reasons are too shallow and too quick.

QUEEN ELIZABETH:
O no, my reasons are too deep and dead;
Too deep and dead, poor infants, in their grave.

KING RICHARD III:
Harp not on that string, madam; that is past.

QUEEN ELIZABETH:
Harp on it still shall I till heart-strings break.

KING RICHARD III:
Now, by my George, my garter, and my crown,--

QUEEN ELIZABETH:
Profaned, dishonour'd, and the third usurp'd.

KING RICHARD III:
I swear--

QUEEN ELIZABETH:
By nothing; for this is no oath:
The George, profaned, hath lost his holy honour;
The garter, blemish'd, pawn'd his knightly virtue;
The crown, usurp'd, disgraced his kingly glory.
if something thou wilt swear to be believed,
Swear then by something that thou hast not wrong'd.

KING RICHARD III:
Now, by the world--

QUEEN ELIZABETH:
'Tis full of thy foul wrongs.

KING RICHARD III:
My father's death--

QUEEN ELIZABETH:
Thy life hath that dishonour'd.

KING RICHARD III:
Then, by myself--

QUEEN ELIZABETH:
Thyself thyself misusest.

KING RICHARD III:
Why then, by God--

QUEEN ELIZABETH:
God's wrong is most of all.
If thou hadst fear'd to break an oath by Him,
The unity the king thy brother made
Had not been broken, nor my brother slain:
If thou hadst fear'd to break an oath by Him,
The imperial metal, circling now thy brow,
Had graced the tender temples of my child,
And both the princes had been breathing here,
Which now, two tender playfellows to dust,
Thy broken faith hath made a prey for worms.
What canst thou swear by now?

KING RICHARD III:
The time to come.

QUEEN ELIZABETH:
That thou hast wronged in the time o'erpast;
For I myself have many tears to wash
Hereafter time, for time past wrong'd by thee.
The children live, whose parents thou hast
slaughter'd,
Ungovern'd youth, to wail it in their age;
The parents live, whose children thou hast butcher'd,
Old wither'd plants, to wail it with their age.
Swear not by time to come; for that thou hast
Misused ere used, by time misused o'erpast.

KING RICHARD III:
As I intend to prosper and repent,
So thrive I in my dangerous attempt
Of hostile arms! myself myself confound!
Heaven and fortune bar me happy hours!
Day, yield me not thy light; nor, night, thy rest!
Be opposite all planets of good luck
To my proceedings, if, with pure heart's love,
Immaculate devotion, holy thoughts,
I tender not thy beauteous princely daughter!
In her consists my happiness and thine;
Without her, follows to this land and me,
To thee, herself, and many a Christian soul,
Death, desolation, ruin and decay:
It cannot be avoided but by this;
It will not be avoided but by this.
Therefore, good mother,--I must can you so--
Be the attorney of my love to her:
Plead what I will be, not what I have been;
Not my deserts, but what I will deserve:
Urge the necessity and state of times,
And be not peevish-fond in great designs.

QUEEN ELIZABETH:
Shall I be tempted of the devil thus?

KING RICHARD III:
Ay, if the devil tempt thee to do good.

QUEEN ELIZABETH:
Shall I forget myself to be myself?

KING RICHARD III:
Ay, if yourself's remembrance wrong yourself.

)",

    R"(QUEEN ELIZABETH:
But thou didst kill my children.

KING RICHARD III:
But in your daughter's womb I bury them:
Where in that nest of spicery they shall breed
Selves of themselves, to your recomforture.

QUEEN ELIZABETH:
Shall I go win my daughter to thy will?

KING RICHARD III:
And be a happy mother by the deed.

QUEEN ELIZABETH:
I go. Write to me very shortly.
And you shall understand from me her mind.

KING RICHARD III:
Bear her my true love's kiss; and so, farewell.
Relenting fool, and shallow, changing woman!
How now! what news?

RATCLIFF:
My gracious sovereign, on the western coast
Rideth a puissant navy; to the shore
Throng many doubtful hollow-hearted friends,
Unarm'd, and unresolved to beat them back:
'Tis thought that Richmond is their admiral;
And there they hull, expecting but the aid
Of Buckingham to welcome them ashore.

KING RICHARD III:
Some light-foot friend post to the Duke of Norfolk:
Ratcliff, thyself, or Catesby; where is he?

CATESBY:
Here, my lord.

KING RICHARD III:
Fly to the duke:
Post thou to Salisbury
When thou comest thither--
Dull, unmindful villain,
Why stand'st thou still, and go'st not to the duke?

CATESBY:
First, mighty sovereign, let me know your mind,
What from your grace I shall deliver to him.

KING RICHARD III:
O, true, good Catesby: bid him levy straight
The greatest strength and power he can make,
And meet me presently at Salisbury.

CATESBY:
I go.

RATCLIFF:
What is't your highness' pleasure I shall do at
Salisbury?

KING RICHARD III:
Why, what wouldst thou do there before I go?

RATCLIFF:
Your highness told me I should post before.

KING RICHARD III:
My mind is changed, sir, my mind is changed.
How now, what news with you?

STANLEY:
None good, my lord, to please you with the hearing;
Nor none so bad, but it may well be told.

KING RICHARD III:
Hoyday, a riddle! neither good nor bad!
Why dost thou run so many mile about,
When thou mayst tell thy tale a nearer way?
Once more, what news?

STANLEY:
Richmond is on the seas.

KING RICHARD III:
There let him sink, and be the seas on him!
White-liver'd runagate, what doth he there?

STANLEY:
I know not, mighty sovereign, but by guess.

KING RICHARD III:
Well, sir, as you guess, as you guess?

STANLEY:
Stirr'd up by Dorset, Buckingham, and Ely,
He makes for England, there to claim the crown.

KING RICHARD III:
Is the chair empty? is the sword unsway'd?
Is the king dead? the empire unpossess'd?
What heir of York is there alive but we?
And who is England's king but great York's heir?
Then, tell me, what doth he upon the sea?

STANLEY:
Unless for that, my liege, I cannot guess.

KING RICHARD III:
Unless for that he comes to be your liege,
You cannot guess wherefore the Welshman comes.
Thou wilt revolt, and fly to him, I fear.

STANLEY:
No, mighty liege; therefore mistrust me not.

KING RICHARD III:
Where is thy power, then, to beat him back?
Where are thy tenants and thy followers?
Are they not now upon the western shore.
Safe-conducting the rebels from their ships!

STANLEY:
No, my good lord, my friends are in the north.

KING RICHARD III:
Cold friends to Richard: what do they in the north,
When they should serve their sovereign in the west?

)",

    R"(STANLEY:
They have not been commanded, mighty sovereign:
Please it your majesty to give me leave,
I'll muster up my friends, and meet your grace
Where and what time your majesty shall please.

KING RICHARD III:
Ay, ay. thou wouldst be gone to join with Richmond:
I will not trust you, sir.

STANLEY:
Most mighty sovereign,
You have no cause to hold my friendship doubtful:
I never was nor never will be false.

KING RICHARD III:
Well,
Go muster men; but, hear you, leave behind
Your son, George Stanley: look your faith be firm.
Or else his head's assurance is but frail.

STANLEY:
So deal with him as I prove true to you.

Messenger:
My gracious sovereign, now in Devonshire,
As I by friends am well advertised,
Sir Edward Courtney, and the haughty prelate
Bishop of Exeter, his brother there,
With many more confederates, are in arms.

Second Messenger:
My liege, in Kent the Guildfords are in arms;
And every hour more competitors
Flock to their aid, and still their power increaseth.

Third Messenger:
My lord, the army of the Duke of Buckingham--

KING RICHARD III:
Out on you, owls! nothing but songs of death?
Take that, until thou bring me better news.

Third Messenger:
The news I have to tell your majesty
Is, that by sudden floods and fall of waters,
Buckingham's army is dispersed and scatter'd;
And he himself wander'd away alone,
No man knows whither.

KING RICHARD III:
I cry thee mercy:
There is my purse to cure that blow of thine.
Hath any well-advised friend proclaim'd
Reward to him that brings the traitor in?

Third Messenger:
Such proclamation hath been made, my liege.

Fourth Messenger:
Sir Thomas Lovel and Lord Marquis Dorset,
'Tis said, my liege, in Yorkshire are in arms.
Yet this good comfort bring I to your grace,
The Breton navy is dispersed by tempest:
Richmond, in Yorkshire, sent out a boat
Unto the shore, to ask those on the banks
If they were his assistants, yea or no;
Who answer'd him, they came from Buckingham.
Upon his party: he, mistrusting them,
Hoisted sail and made away for Brittany.

KING RICHARD III:
March on, march on, since we are up in arms;
If not to fight with foreign enemies,
Yet to beat down these rebels here at home.

CATESBY:
My liege, the Duke of Buckingham is taken;
That is the best news: that the Earl of Richmond
Is with a mighty power landed at Milford,
Is colder tidings, yet they must be told.

KING RICHARD III:
Away towards Salisbury! while we reason here,
A royal battle might be won and lost
Some one take order Buckingham be brought
To Salisbury; the rest march on with me.

DERBY:
Sir Christopher, tell Richmond this from me:
That in the sty of this most bloody boar
My son George Stanley is frank'd up in hold:
If I revolt, off goes young George's head;
The fear of that withholds my present aid.
But, tell me, where is princely Richmond now?

CHRISTOPHER:
At Pembroke, or at Harford-west, in Wales.

DERBY:
What men of name resort to him?

CHRISTOPHER:
Sir Walter Herbert, a renowned soldier;
Sir Gilbert Talbot, Sir William Stanley;
Oxford, redoubted Pembroke, Sir James Blunt,
And Rice ap Thomas with a valiant crew;
And many more of noble fame and worth:
And towards London they do bend their course,
If by the way they be not fought withal.

DERBY:
Return unto thy lord; commend me to him:
Tell him the queen hath heartily consented
He shall espouse Elizabeth her daughter.
These letters will resolve him of my mind. Farewell.

BUCKINGHAM:
Will not King Richard let me speak with him?

Sheriff:
No, my good lord; therefore be patient.

BUCKINGHAM:
Hastings, and Edward's children, Rivers, Grey,
Holy King Henry, and thy fair son Edward,
Vaughan, and all that have miscarried
By underhand corrupted foul injustice,
If that your moody discontented souls
Do through the clouds behold this present hour,
Even for revenge mock my destruction!
This is All-Souls' day, fellows, is it not?

Sheriff:
It is, my lord.

BUCKINGHAM:
Why, then All-Souls' day is my body's doomsday.
This is the day that, in King Edward's time,
I wish't might fall on me, when I was found
False to his children or his wife's allies
This is the day wherein I wish'd to fall
By the false faith of him I trusted most;
This, this All-Souls' day to my fearful soul
Is the determined respite of my wrongs:
That high All-Seer that I dallied with
Hath turn'd my feigned prayer on my head
And given in earnest what I begg'd in jest.
Thus doth he force the swords of wicked men
To turn their own points on their masters' bosoms:
Now Margaret's curse is fallen upon my head;
'When he,' quoth she, 'shall split thy heart with sorrow,
Remember Margaret was a prophetess.'
Come, sirs, convey me to the block of shame;
Wrong hath but wrong, and blame the due of blame.

)",

    R"(RICHMOND:
Fellows in arms, and my most loving friends,
Bruised underneath the yoke of tyranny,
Thus far into the bowels of the land
Have we march'd on without impediment;
And here receive we from our father Stanley
Lines of fair comfort and encouragement.
The wretched, bloody, and usurping boar,
That spoil'd your summer fields and fruitful vines,
Swills your warm blood like wash, and makes his trough
In your embowell'd bosoms, this foul swine
Lies now even in the centre of this isle,
Near to the town of Leicester, as we learn
From Tamworth thither is but one day's march.
In God's name, cheerly on, courageous friends,
To reap the harvest of perpetual peace
By this one bloody trial of sharp war.

OXFORD:
Every man's conscience is a thousand swords,
To fight against that bloody homicide.

HERBERT:
I doubt not but his friends will fly to us.

BLUNT:
He hath no friends but who are friends for fear.
Which in his greatest need will shrink from him.

RICHMOND:
All for our vantage. Then, in God's name, march:
True hope is swift, and flies with swallow's wings:
Kings it makes gods, and meaner creatures kings.

KING RICHARD III:
Here pitch our tents, even here in Bosworth field.
My Lord of Surrey, why look you so sad?

SURREY:
My heart is ten times lighter than my looks.

KING RICHARD III:
My Lord of Norfolk,--

NORFOLK:
Here, most gracious liege.

KING RICHARD III:
Norfolk, we must have knocks; ha! must we not?

NORFOLK:
We must both give and take, my gracious lord.

KING RICHARD III:
Up with my tent there! here will I lie tonight;
But where to-morrow?  Well, all's one for that.
Who hath descried the number of the foe?

NORFOLK:
Six or seven thousand is their utmost power.

KING RICHARD III:
Why, our battalion trebles that account:
Besides, the king's name is a tower of strength,
Which they upon the adverse party want.
Up with my tent there! Valiant gentlemen,
Let us survey the vantage of the field
Call for some men of sound direction
Let's want no discipline, make no delay,
For, lords, to-morrow is a busy day.

RICHMOND:
The weary sun hath made a golden set,
And by the bright track of his fiery car,
Gives signal, of a goodly day to-morrow.
Sir William Brandon, you shall bear my standard.
Give me some ink and paper in my tent
I'll draw the form and model of our battle,
Limit each leader to his several charge,
And part in just proportion our small strength.
My Lord of Oxford, you, Sir William Brandon,
And you, Sir Walter Herbert, stay with me.
The Earl of Pembroke keeps his regiment:
Good Captain Blunt, bear my good night to him
And by the second hour in the morning
Desire the earl to see me in my tent:
Yet one thing more, good Blunt, before thou go'st,
Where is Lord Stanley quarter'd, dost thou know?

BLUNT:
Unless I have mista'en his colours much,
Which well I am assured I have not done,
His regiment lies half a mile at least
South from the mighty power of the king.

RICHMOND:
If without peril it be possible,
Good Captain Blunt, bear my good-night to him,
And give him from me this most needful scroll.

BLUNT:
Upon my life, my lord, I'll under-take it;
And so, God give you quiet rest to-night!

RICHMOND:
Good night, good Captain Blunt. Come gentlemen,
Let us consult upon to-morrow's business
In to our tent; the air is raw and cold.

KING RICHARD III:
What is't o'clock?

CATESBY:
It's supper-time, my lord;
It's nine o'clock.

KING RICHARD III:
I will not sup to-night.
Give me some ink and paper.
What, is my beaver easier than it was?
And all my armour laid into my tent?

CATESBY:
If is, my liege; and all things are in readiness.

KING RICHARD III:
Good Norfolk, hie thee to thy charge;
Use careful watch, choose trusty sentinels.

NORFOLK:
I go, my lord.

KING RICHARD III:
Stir with the lark to-morrow, gentle Norfolk.

NORFOLK:
I warrant you, my lord.

KING RICHARD III:
Catesby!

CATESBY:
My lord?

KING RICHARD III:
Send out a pursuivant at arms
To Stanley's regiment; bid him bring his power
Before sunrising, lest his son George fall
Into the blind cave of eternal night.
Fill me a bowl of wine. Give me a watch.
Saddle white Surrey for the field to-morrow.
Look that my staves be sound, and not too heavy.
Ratcliff!

RATCLIFF:
My lord?

KING RICHARD III:
Saw'st thou the melancholy Lord Northumberland?

)"
};

// Full text for training
const std::string shakespeare_text = concatenateTexts(shakespeare_text_parts);

// Prompt for text generation
const std::string shakespeare_prompt = R"(QUEEN ELIZABETH:
But thou didst kill my children.

KING RICHARD III:
But in your daughter's womb I bury them:
Where in that nest of spicery they shall breed
Selves of themselves, to your recomforture.

QUEEN ELIZABETH:
Shall I go win my daughter to thy will?

KING RICHARD III:
And be a happy mother by the deed.

QUEEN ELIZABETH:
I go. Write to me very shortly.
And you shall understand from me her mind.

)";

#endif // SlmData_H